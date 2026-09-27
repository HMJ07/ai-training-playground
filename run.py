"""Entry point: `python run.py` starts the local server and opens the UI.

Everything runs on your machine; no data leaves localhost.
"""

import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def _open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Timer(1.5, _open_browser).start()
    uvicorn.run("aiengine.server.app:app", host=HOST, port=PORT, reload=False)
