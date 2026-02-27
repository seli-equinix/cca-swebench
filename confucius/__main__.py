# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""CCA HTTP Server entry point.

Usage:
    confucius --port 8500
    python -m confucius --port 8500
"""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="CCA HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8500, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import uvicorn

    from confucius.server.app import app

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
