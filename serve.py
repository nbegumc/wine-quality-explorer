"""Serve the dashboard locally without any third-party server dependency."""
import functools
import http.server
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "dist"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler)
    print("Open http://localhost:8000 — press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
