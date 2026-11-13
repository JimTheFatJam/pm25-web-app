#!/usr/bin/env bash
#
# Starts the app under gunicorn, with the environment loaded from a file that
# lives OUTSIDE this repository (it holds the Google Cloud project id and the
# path to the Earth Engine service-account key).
#
# Used by PM2 on hosts without root:
#
#     pm2 start deploy/run_gunicorn.sh --name pm25 --interpreter bash
#
# and equally runnable by hand for debugging:
#
#     PM25_ENV_FILE=~/pm25-secrets/pm25-web.env deploy/run_gunicorn.sh
#
# On a host where you DO have root, prefer the systemd unit next to this file;
# it does the same thing with proper supervision.

set -euo pipefail

APP_DIR="${PM25_APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${PM25_ENV_FILE:-$HOME/pm25-secrets/pm25-web.env}"
PORT="${PM25_PORT:-20451}"
BIND="${PM25_BIND:-127.0.0.1}"

cd "$APP_DIR"

if [ -f "$ENV_FILE" ]; then
    # `set -a` exports everything the file defines, so gunicorn's children
    # (and therefore config/settings.py) see it.
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
else
    echo "WARNING: no env file at $ENV_FILE -- starting without PM25_PUBLIC_MODE" >&2
    echo "         and without Earth Engine credentials. Predictions will fail." >&2
fi

# ONE worker, deliberately. The job registry, the mobile (cell, T*) cache, the
# SHAP feature store and the Earth Engine availability state are all
# in-process module-level state: with two workers a status poll could land on
# a worker that never ran the job. Concurrency comes from threads, which suits
# work that is I/O-bound on Earth Engine anyway.
#
# --timeout 300 covers the synchronous mobile endpoint on a cold cell
# (measured 15-25 s, with headroom). The web prediction endpoint returns a job
# id immediately and runs in a background thread, so it is unaffected.
exec "$APP_DIR/.venv/bin/gunicorn" \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --graceful-timeout 30 \
    --bind "${BIND}:${PORT}" \
    --access-logfile - \
    --error-logfile - \
    backend.app:app
