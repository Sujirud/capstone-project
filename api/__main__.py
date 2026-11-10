"""Start the api on 127.0.0.1 with uvicorn, the same as the uvicorn command in api/main.py.

    uv run python -m api
    uv run python -m api --port 8001

The host is always 127.0.0.1, so other machines on the network cannot reach the api
(api.md section 6); only the port can change. The files are checked before uvicorn
starts, so a missing one stops with the command to run instead of a traceback. At start
the api also warns when web/dist is missing or older than the newest file in web/src,
web/index.html, shared/quality.json or shared/i18n/ (api/main.py web_build_warning), with
the command that builds it again.
"""

import argparse
import sys

import uvicorn

from .engine import StartupError, load_parts

HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m api", description=f"Start the api on {HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error(f"--port must be 1-65535, got {args.port}")
    try:
        load_parts()
    except StartupError as e:
        print(f"Cannot start the api: {e}", file=sys.stderr)
        return 1
    from .main import app

    uvicorn.run(app, host=HOST, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
