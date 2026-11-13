# Deploying to https://pm25.dcism.org/

Two supported shapes:

| | **A. Dedicated VM, root** | **B. Shared host, no root** |
|---|---|---|
| Web server | nginx + TLS you configure | Apache the department runs |
| Process supervision | systemd unit | PM2 (or any user process manager) |
| App location | `/srv/pm25-web-app` | `~/pm25-web-app` |
| Entry point | `deploy/pm25-web.service` | `deploy/run_gunicorn.sh` |

**pm25.dcism.org runs shape B.** Sections 1 and 2 (secrets, abuse protection)
apply to both; section 3 covers A and section 3b covers B.

Everything in section 3 assumes the app lives at `/srv/pm25-web-app` and runs
as the unprivileged user `pm25`.

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

1. In Google Cloud, create a service account in the same project (e.g.
   `pm25-web-server@<project>.iam.gserviceaccount.com`).
2. Grant it **Earth Engine Resource Writer** and **Service Usage Consumer**.
   Not Owner, not Editor.
3. Create a JSON key and register the address at
   <https://signup.earthengine.google.com/#!/service_accounts>.
4. Put it somewhere outside the repository, readable only by the service
   user — `chmod 600`, in a `chmod 700` directory — and point
   `GOOGLE_APPLICATION_CREDENTIALS` at it.

> Note: `ee.Initialize()` does **not** read `GOOGLE_APPLICATION_CREDENTIALS`.
> It calls `get_persistent_credentials()`, which only looks for the file
> `earthengine authenticate` writes, and otherwise fails with *"Please
> authorize access to your Earth Engine account"*. `backend/app.py` therefore
> reads the variable itself and builds `ee.ServiceAccountCredentials`
> explicitly. Unset the variable and it falls back to interactive
> credentials, which is how a developer machine works.

---

## 2. Abuse protection

### What is actually at risk

Not CPU — **Earth Engine quota**, which is attached to a billable Google Cloud
project. One prediction job issues many `reduceRegions` calls. The full grid
is 313,845 cells, so a script walking it would mint a fresh `(cell, T*)` cache
key every time and never hit a cache. That is the attack to stop.

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

**Third-party datasets.** The WorldPop rasters and the Natural Earth coastline
**are committed** (~19 MB): `static_predictors.py` reads them on *every*
prediction, because population and distance-to-coast are model features. A
deployment without them starts fine and then fails each request with
`StaticPredictorError`.

Only the OSM extract (~77 MB) is excluded, and it is genuinely build-time:
road density is precomputed into `data/static_layers/road_density_by_cell.csv`,
which is committed. You need the `.pbf` only to rebuild that cache — see
`data/local_datasets/osm/README.txt`.

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

## 3b. Shared host without root (how pm25.dcism.org runs)

No `sudo`, Apache already serving `~/<domain>/` as a document root, and
`mod_proxy` + `mod_rewrite` enabled. The app runs as your own user on a high
port and Apache reverse-proxies to it from a `.htaccess` you control.

### Install

```bash
git clone https://github.com/JimTheFatJam/pm25-web-app.git ~/pm25-web-app
cd ~/pm25-web-app && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build_full_grid.py          # generates the 88 MB grid, ~40 s
```

### Secrets, outside the repository

```bash
mkdir -p ~/pm25-secrets && chmod 700 ~/pm25-secrets
```

Put the Earth Engine service-account key at
`~/pm25-secrets/gee-service-account.json` (`chmod 600`), then create
`~/pm25-secrets/pm25-web.env` from `deploy/pm25-web.env.example`, with
`HOME` left alone and the credentials path pointing into `~/pm25-secrets/`.

### Run it

`deploy/run_gunicorn.sh` loads that env file and execs gunicorn with a single
worker. Under PM2:

```bash
cd ~/pm25-web-app && pm2 start deploy/run_gunicorn.sh --name pm25 --interpreter bash && pm2 save
```

`PM25_PORT` (default 20451) picks the port; it binds to 127.0.0.1 only, so the
app is never reachable except through Apache.

### Expose it

In the document root — **not** the app directory:

```apache
RewriteEngine On
RewriteRule ^(.*)$ http://127.0.0.1:20451/$1 [P,L]
```

Confirm with `curl -sS https://<domain>/api/health`.

### Surviving a reboot

`pm2 startup` needs root. Without it, use cron:

```bash
(crontab -l 2>/dev/null; echo "@reboot sleep 30 && $HOME/.local/bin/pm2 resurrect") | crontab -
```

`pm2 save` records the current process list; `pm2 resurrect` restores it.

### Updating

`deploy/deploy.sh` assumes systemd. On a shared host:

```bash
cd ~/pm25-web-app && git pull --ff-only && .venv/bin/pip install -q -r requirements.txt && pm2 restart pm25 && sleep 45 && curl -sS http://127.0.0.1:20451/api/health
```

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
