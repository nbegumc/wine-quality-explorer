"""Serve the dashboard locally without any third-party server dependency.

HOST and PORT can be set in the environment; the Docker image binds 0.0.0.0.
"""
import functools
import http.server
import os
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "app"
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    print(f"Open http://localhost:{port} — press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
