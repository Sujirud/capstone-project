"""The api of api.md: FastAPI on 127.0.0.1 that measures a frontal and a profile image.

Run from the project root, always as api.main, because the code here imports itself
relatively:

    uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
    uv run python -m api

This file stays light: uvicorn and the scripts import the package without loading any model.
"""

API_VERSION = "0.1.0"
