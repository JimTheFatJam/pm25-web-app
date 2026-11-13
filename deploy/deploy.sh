#!/usr/bin/env bash
#
# Pull the latest pushed changes and restart the service, with a health
# check and automatic rollback.
#
#   sudo -u pm25 /srv/pm25-web-app/deploy/deploy.sh
#
# Safe to re-run. Does nothing destructive to data/ -- the grid, the model
# artifacts and the local datasets are left alone unless they are missing.

set -euo pipefail

APP_DIR="${PM25_APP_DIR:-/srv/pm25-web-app}"
SERVICE="${PM25_SERVICE:-pm25-web}"
HEALTH_URL="${PM25_HEALTH_URL:-http://127.0.0.1:5050/api/health}"
BRANCH="${PM25_BRANCH:-main}"

cd "$APP_DIR"

echo "==> Current commit: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
PREVIOUS="$(git rev-parse HEAD)"

echo "==> Fetching origin/$BRANCH"
git fetch --prune origin "$BRANCH"

if [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "==> Already up to date. Nothing to deploy."
    exit 0
fi

echo "==> Changes to apply:"
git --no-pager log --oneline HEAD.."origin/$BRANCH" | sed 's/^/    /'

# --ff-only: refuse to deploy if the server has local commits. Fix those
# deliberately rather than letting a merge happen unattended.
git merge --ff-only "origin/$BRANCH"
echo "==> Now at: $(git rev-parse --short HEAD)"

echo "==> Installing dependencies"
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r requirements.txt

# The 1 km grid (~88 MB) is generated, not committed -- build it if this is
# a first deploy or if it was removed. Takes a few minutes.
if [ ! -f "$APP_DIR/data/official_grid_ph_full.gpkg" ]; then
    echo "==> Grid missing; generating with build_full_grid.py (this takes a few minutes)"
    "$APP_DIR/.venv/bin/python" build_full_grid.py
fi

echo "==> Restarting $SERVICE"
sudo systemctl restart "$SERVICE"

echo -n "==> Waiting for health"
for i in $(seq 1 60); do
    if curl -fsS --max-time 5 "$HEALTH_URL" > /tmp/pm25-health.json 2>/dev/null; then
        if grep -q '"model_loaded": *true' /tmp/pm25-health.json; then
            echo " OK"
            echo "==> Health:"
            sed 's/^/    /' /tmp/pm25-health.json
            echo "==> Deployed $(git rev-parse --short HEAD) successfully."
            exit 0
        fi
    fi
    echo -n "."
    sleep 3
done

echo " FAILED"
echo "==> Service did not become healthy. Rolling back to $PREVIOUS"
git reset --hard "$PREVIOUS"
"$APP_DIR/.venv/bin/pip" install --quiet -r requirements.txt || true
sudo systemctl restart "$SERVICE"
echo "==> Rolled back. Inspect: sudo journalctl -u $SERVICE -n 100 --no-pager"
exit 1
