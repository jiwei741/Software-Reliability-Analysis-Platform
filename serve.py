import argparse
import http.server
import socket
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

def pick_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the reliability platform locally")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).parent,
        help="Root directory to serve"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Only start the server without opening a browser"
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.exists():
        raise SystemExit(f"Directory {directory} does not exist")

    port = pick_port(args.port)
    handler_cls = partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.ThreadingTCPServer(("", port), handler_cls) as httpd:
        url = f"http://127.0.0.1:{port}/index.html"
        print(f"Serving {directory} at {url}")
        if not args.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server...")


if __name__ == "__main__":
    main()
