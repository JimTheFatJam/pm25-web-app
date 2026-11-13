"""
run_app.py -- Launcher for the PM2.5 Prediction Dashboard.

Starts the local Flask backend and opens the dashboard in your default
browser. This is a genuinely local, offline-capable application: the map
is a pre-shipped vector boundary (no tile server), D3.js is vendored
locally (no CDN), and all spatial logic runs in the Flask backend against
data files already in this project. No internet connection is required
after installation.
"""

import os
import webbrowser
import threading
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app import app

PORT = 5050
URL = f"http://127.0.0.1:{PORT}"


def open_browser():
    time.sleep(1.5)  # give Flask a moment to bind before opening the browser
    webbrowser.open(URL)


if __name__ == "__main__":
    # Local development only. The public deployment runs gunicorn behind
    # nginx (see DEPLOYMENT.md) -- Flask's built-in server is single-
    # threaded-ish, has no timeouts and is not hardened for the internet.
    if os.environ.get("PM25_PUBLIC_MODE", "").strip() in ("1", "true", "yes", "on"):
        sys.exit("run_app.py is the LOCAL development launcher. PM25_PUBLIC_MODE is set, "
                 "so start the service with gunicorn instead -- see DEPLOYMENT.md.")
    print(f"Starting PM2.5 Prediction Dashboard at {URL}")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
