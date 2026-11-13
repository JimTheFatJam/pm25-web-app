# Deploying to https://pm25.dcism.org/

Production setup for the public instance: Ubuntu/Debian VM with root, nginx +
TLS in front, gunicorn running the Flask app as a systemd service.

Everything below assumes the app lives at `/srv/pm25-web-app` and runs as the
unprivileged user `pm25`.

---

## 1. What must never be committed

The repository is public. Three things stay off it, and `.gitignore` enforces
the first two:

| Secret | Where it lives | How the app finds it |
|---|---|---|
| Google Cloud project id | `/etc/pm25-web/pm25-web.env` | `PM25_GEE_PROJECT_ID` |
| Earth Engine service-account key | `/etc/pm25-web/gee-service-account.json`, mode `600` | `GOOGLE_APPLICATION_CREDENTIALS` |
| Your local dev project id | `config/local_settings.py` | gitignored; overrides `settings.py` |

`config/settings.py` reads the project id from the environment and falls back
to the gitignored `config/local_settings.py`, so nothing identifying is in git.
There are no other API keys in this project — Earth Engine is the only external
service, and it authenticates with Google credentials rather than a key string.

> Reverse geocoding in the mobile app calls OpenStreetMap Nominatim, which
> needs no key at all.

### Earth Engine on a headless server

Interactive `earthengine authenticate` is not suitable for a server. Use a
service account instead:

1. In Google Cloud, create a service account in the same project and download
   a JSON key.
2. Register it with Earth Engine at
   <https://signup.earthengine.google.com/#!/service_accounts>.
3. Install it and lock it down:

```bash
sudo install -D -o pm25 -g pm25 -m 600 gee-service-account.json /etc/pm25-web/gee-service-account.json
```

`ee.Initialize()` picks it up as Application Default Credentials via
`GOOGLE_APPLICATION_CREDENTIALS` — no code change needed.

---

## 2. Abuse protection

### What is actually at risk

Not CPU — **Earth Engine quota**, which is attached to a billable Google Cloud
project. One prediction job issues many `reduceRegions` calls. The grid has
~62,000 cells, so a script walking it would mint a fresh `(cell, T*)` cache key
every time and never hit a cache. That is the attack to stop.

### Layers, cheapest first

| Layer | Where | Default |
|---|---|---|
| Per-IP rate limit (shared, pre-Python) | `deploy/nginx-pm25-limits.conf` | 10/min predict, 30/min mobile, 120/min general |
| Per-IP rate limit (per endpoint, precise) | `backend/limits.py` | 5/h + 20/day predict · 30/h + 200/day mobile · 600/h read · 1200/h poll |
| Global concurrency cap | `backend/limits.py` | 2 simultaneous prediction jobs |
| Global daily GEE budget | `backend/limits.py` | 300 compute jobs/day, then HTTP 503 until UTC midnight |
| Request-size cap | `backend/limits.py` | 2,000 cells/request, 2 MB body |
| Security headers + strict CSP | `backend/app.py` | nosniff, DENY framing, no-referrer, self-only CSP |

All of it is **inert unless `PM25_PUBLIC_MODE=1`**, so local development and the
test suite are unaffected. Tune any limit through the environment without
editing code — see `deploy/pm25-web.env.example`.

Cached mobile readings cost nothing, so only genuine compute is charged against
the daily budget: a popular cell serves thousands of visitors for one budget
unit.

To see the current state at any time:

```bash
curl -s https://pm25.dcism.org/api/health | python3 -m json.tool
```

### Why exactly one worker process

`--workers 1 --threads 8` in the systemd unit is **not** arbitrary. The job
registry, the mobile `(cell, T*)` cache, the SHAP feature store and the GEE
availability state are all in-process module-level state. With two workers, a
status poll could hit a worker that never ran the job, and the caches would
silently diverge. Concurrency comes from threads, which is right anyway — the
work is I/O-bound waiting on Earth Engine.

### Still worth doing

- Set a **budget alert and a hard cap** on the Google Cloud project. The
  in-app budget protects against the expected failure modes; a billing cap
  protects against the unexpected ones.
- Put the site behind Cloudflare (free tier) if it is ever linked publicly, for
  volumetric protection the app cannot provide by itself.

---

## 3. First-time server setup

```bash
sudo adduser --system --group --home /srv/pm25-web-app pm25
sudo apt update && sudo apt install -y python3-venv python3-dev build-essential \
    libgdal-dev gdal-bin nginx git curl
```

```bash
sudo -u pm25 git clone https://github.com/JimTheFatJam/pm25-web-app.git /srv/pm25-web-app
```

```bash
cd /srv/pm25-web-app && sudo -u pm25 python3 -m venv .venv && sudo -u pm25 .venv/bin/pip install -r requirements.txt
```

### Data the repository does not carry

Two things are excluded from git because they are large and reproducible:

**The 1 km grid (~88 MB)** — generated, not downloaded:

```bash
sudo -u pm25 /srv/pm25-web-app/.venv/bin/python /srv/pm25-web-app/build_full_grid.py
```

**Third-party datasets (~99 MB)** — only needed to *rebuild* static layers, not
to serve predictions. `data/static_layers/road_density_by_cell.csv` is committed,
so a normal deployment does not need them. If you do need them, each folder's
`README.txt` names the exact file and source (OSM extract, Natural Earth
coastline, WorldPop).

### Config, service, TLS

```bash
sudo install -D -o root -g pm25 -m 640 /srv/pm25-web-app/deploy/pm25-web.env.example /etc/pm25-web/pm25-web.env
```

Edit that file: set `PM25_GEE_PROJECT_ID` and the credentials path. Then:

```bash
sudo cp /srv/pm25-web-app/deploy/pm25-web.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now pm25-web
```

```bash
sudo cp /srv/pm25-web-app/deploy/nginx-pm25-limits.conf /etc/nginx/conf.d/pm25-limits.conf && sudo cp /srv/pm25-web-app/deploy/nginx-pm25.conf /etc/nginx/sites-available/pm25 && sudo ln -sf /etc/nginx/sites-available/pm25 /etc/nginx/sites-enabled/pm25
```

```bash
sudo certbot --nginx -d pm25.dcism.org && sudo nginx -t && sudo systemctl reload nginx
```

Verify:

```bash
curl -s https://pm25.dcism.org/api/health | python3 -m json.tool
```

`model_loaded: true` and `availability_status: "ready"` means it is serving.
The availability check runs in the background and takes ~30 s after a restart.

---

## 4. Updating the site from GitHub

This is the whole point of the repository: push locally, pull on the server.

```bash
sudo -u pm25 /srv/pm25-web-app/deploy/deploy.sh
```

That script:

1. shows the commits about to be applied,
2. `git merge --ff-only` (refuses if the server has diverging local commits —
   fix those deliberately rather than auto-merging),
3. reinstalls `requirements.txt`,
4. regenerates the grid if it is missing,
5. restarts the service,
6. polls `/api/health` for up to 3 minutes, and
7. **rolls back to the previous commit and restarts** if it never becomes
   healthy.

Exit code 0 means the new version is live. Nothing in `data/` is touched.

### If you want it automatic

A pull every 10 minutes, only acting when something changed:

```bash
sudo systemd-run --on-calendar='*:0/10' --unit=pm25-autodeploy --uid=pm25 /srv/pm25-web-app/deploy/deploy.sh
```

For push-triggered deploys instead, add a GitHub webhook pointing at a small
listener, or a self-hosted runner. The pull model above is simpler and has no
inbound attack surface — prefer it unless you need instant deploys.

### Model updates

Model artifacts (~18 MB) **are** committed, so swapping the deployed ensemble is
an ordinary commit: replace the files in `data/models/<set>/`, repoint
`MODEL_DIR` in `config/settings.py`, push, and run `deploy.sh`. The startup
validation in `backend/prediction/model_loader.py` refuses to serve a
mismatched artifact set, and `deploy.sh` rolls back when that happens.

---

## 5. Operations

```bash
sudo journalctl -u pm25-web -f
```

```bash
sudo systemctl restart pm25-web
```

Restarting clears all in-memory caches (jobs, mobile readings, SHAP rows). That
is harmless — the next request recomputes — but the first mobile request per
cell afterwards is slow again (~15–25 s).

Log rotation for nginx is handled by the distribution's default logrotate
config. The app itself logs to the journal, which systemd rotates.
