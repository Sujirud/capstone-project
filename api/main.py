"""The FastAPI app: GET /health, POST /analyze and the built web page (api.md, web.md section 8.3).

Run from the project root, always as api.main (the code in api/ imports itself relatively):

    uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
    uv run python -m api

- Start: the lifespan builds the Engine once (engine.py); anything missing stops the
  api with the command to run. MultiPartParser.spool_max_size is raised to
  request.max_bytes of shared/quality.json, so an upload stays in memory and never goes
  to a temporary file. uvicorn's access log is turned off.
- RequestGuard, a plain ASGI middleware: a uuid4 request_id for every request, 413 when
  Content-Length is over request.max_bytes before the body is read, or when a body sent
  without Content-Length (chunked) grows over it while it is read; Cache-Control:
  no-store on every answer; one log line per request with only request_id, the time
  taken, the HTTP status and the error code (api.md section 6); and a 500 for anything
  unexpected, with no traceback in the log, since a message may hold values.
- Errors have the body {"error": code, "request_id": ...} and nothing else. FastAPI's
  own 422 would echo what was sent, so it is replaced. A body that does not parse as
  multipart (Starlette's 400) is invalid_request (422), because 400 is decode_failed;
  other HTTP errors, such as 404, keep their status with invalid_request.
- POST /analyze is a plain def, so FastAPI runs it in its thread pool, and one lock
  makes requests run one at a time (api.md section 3). The images are decoded with
  image_io (415, 400, 413) and dropped as soon as the result is built; nothing is cached.
- /docs and /redoc are off because they load files from a CDN; /openapi.json stays, with
  the schema of meta added as components.schemas.Meta for the web's types.
- web_dist (config.toml) is mounted at / after every endpoint when it exists. At start
  the api warns when it is missing, or older than the newest file the build is made from
  (web/src, web/index.html, shared/quality.json and shared/i18n/), so the page served
  would not be the code on disk.
"""

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from facemeasure import i18n, image_io, paths, quality, response

from . import API_VERSION, schemas
from .engine import Engine, StartupError

LOGGER_NAME = "api"
log = logging.getLogger(LOGGER_NAME)

REQUEST_ID = "request_id"  # keys of the request state that RequestGuard reads back
ERROR = "error"
NO_STORE = "no-store"
WEB_MOUNT = "web"
ERROR_RESPONSES = {schemas.ERROR_STATUS[c]: {"model": schemas.ErrorBody, "description": c} for c in schemas.ERROR_CODES}


class ApiError(Exception):
    """An error answer with one of the codes of schemas.ERROR_STATUS."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


QUIET_LOGGERS = ("python_multipart",)  # logs details of the body it parses, which must not reach the log


def setup_logging() -> None:
    """The api logger writes 'LEVEL: message' lines to stderr, next to uvicorn's own; others stay quiet."""
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def error_body(request_id: str, code: str) -> dict:
    return {"error": code, "request_id": request_id}


# ---------------------------------------------------------------- the guard


def _content_length(scope: Scope) -> int | None:
    value = Headers(scope=scope).get("content-length")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class _Exchange:
    """One request seen by RequestGuard: the bytes received, the status sent, and whether the guard answered."""

    def __init__(self, receive: Receive, send: Send, state: dict, limit: int):
        self._receive = receive
        self._send = send
        self.state = state
        self.limit = limit
        self.received = 0
        self.status: int | None = None
        self.answered = False  # the guard sent its own answer; anything the app sends after is dropped

    async def answer(self, code: str) -> None:
        """Send an error answer, unless an answer has started already."""
        self.state.setdefault(ERROR, code)
        if self.status is not None:
            return
        self.answered = True
        self.status = schemas.ERROR_STATUS[code]
        body = json.dumps(error_body(self.state[REQUEST_ID], code)).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", NO_STORE.encode("ascii")),
        ]
        await self._send({"type": "http.response.start", "status": self.status, "headers": headers})
        await self._send({"type": "http.response.body", "body": body})

    async def receive(self) -> Message:
        if self.answered:
            return {"type": "http.disconnect"}
        message = await self._receive()
        if message["type"] == "http.request":
            self.received += len(message.get("body", b""))
            if self.received > self.limit:
                await self.answer(schemas.TOO_LARGE)
                return {"type": "http.disconnect"}
        return message

    async def send(self, message: Message) -> None:
        if self.answered:
            return
        if message["type"] == "http.response.start":
            self.status = message["status"]
            message.setdefault("headers", [])
            MutableHeaders(scope=message)["Cache-Control"] = NO_STORE
        await self._send(message)


class RequestGuard:
    """request_id, the size limit, Cache-Control, the one log line and the last-resort 500 (see the module text)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        state = {**(scope.get("state") or {}), REQUEST_ID: str(uuid.uuid4())}
        scope = {**scope, "state": state}
        ex = _Exchange(receive, send, state, scope["app"].state.max_bytes)
        try:
            length = _content_length(scope)
            if length is not None and length > ex.limit:
                await ex.answer(schemas.TOO_LARGE)
            else:
                await self.app(scope, ex.receive, ex.send)
        except Exception:  # noqa: BLE001  the error code is the answer; the message may hold values, so it is not logged
            await ex.answer(schemas.INTERNAL_ERROR)
        finally:
            ms = round((time.perf_counter() - start) * 1000)
            log.info("request_id=%s status=%s duration_ms=%d error=%s", state[REQUEST_ID], ex.status, ms, state.get(ERROR, "-"))


# ---------------------------------------------------------------- the app


def _error(request: Request, code: str, status: int | None = None, headers=None) -> JSONResponse:
    state = request.scope["state"]
    state.setdefault(ERROR, code)
    return JSONResponse(error_body(state[REQUEST_ID], code), status_code=status or schemas.ERROR_STATUS[code], headers=headers)


def parse_meta(text: str) -> schemas.Meta:
    try:
        return schemas.Meta.model_validate_json(text)
    except ValidationError:
        raise ApiError(schemas.INVALID_REQUEST) from None


def read_upload(upload: UploadFile | None) -> bytes:
    """The bytes of an upload, b"" for none; the spooled file is closed at once."""
    if upload is None:
        return b""
    try:
        return upload.file.read()
    finally:
        upload.file.close()


def decode(data: bytes, max_pixels: int):
    """image_io.read_image with its errors as the api's error codes."""
    try:
        return image_io.read_image(data, max_pixels=max_pixels)
    except image_io.UnsupportedImageType:
        raise ApiError(schemas.UNSUPPORTED_MEDIA_TYPE) from None
    except image_io.ImageTooLarge:
        raise ApiError(schemas.TOO_LARGE) from None
    except image_io.ImageDecodeError:
        raise ApiError(schemas.DECODE_FAILED) from None


# ---------------------------------------------------------------- the web build

BUILD_COMMAND = "npm --prefix web run build"
WEB_INDEX = "index.html"  # Vite writes it on every build, so its time is the time of the build


def web_sources() -> list[Path]:
    """What the web build is made from: web/src, web/index.html, and the files of shared/ it includes."""
    shared = paths.path("shared")
    return [paths.path("web_src"), paths.path("web_index"), shared / quality.QUALITY_FILE, shared / i18n.FOLDER_NAME]


def _newest_file(sources: list[Path]) -> tuple[float, Path] | None:
    newest = None
    for source in sources:
        files = [source] if source.is_file() else (p for p in source.rglob("*") if p.is_file())
        for f in files:
            stamp = f.stat().st_mtime
            if newest is None or stamp > newest[0]:
                newest = (stamp, f)
    return newest


def web_build_warning(web_dist: Path, sources: list[Path]) -> str | None:
    """Why the web build should be built again, or None when it is newer than every source file."""
    index = Path(web_dist) / WEB_INDEX
    if not index.is_file():
        return f"the web page is not built (no {index}), so / serves nothing; build it with: {BUILD_COMMAND}"
    newest = _newest_file(sources)
    if newest is not None and newest[0] > index.stat().st_mtime:
        return f"the web build in {web_dist} is older than {newest[1]}, changed after the last build; build it again with: {BUILD_COMMAND}"
    return None


def create_app(
    engine_factory: Callable[[], Engine] = Engine.load,
    web_dist: Path | None = None,
    max_pixels: int = image_io.DEFAULT_MAX_PIXELS,
    sources: list[Path] | None = None,
) -> FastAPI:
    """The api; tests pass a fake engine, another web folder or its sources, or a lower pixel limit."""
    setup_logging()
    web = Path(web_dist) if web_dist is not None else paths.path("web_dist")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.getLogger("uvicorn.access").disabled = True
        try:
            engine = engine_factory()
        except StartupError as e:
            log.error("cannot start the api: %s", e)
            raise
        app.state.engine = engine
        app.state.max_bytes = engine.config.request.max_bytes
        MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, app.state.max_bytes)
        for message in engine.warnings:
            log.warning("%s", message)
        stale = web_build_warning(web, sources if sources is not None else web_sources())
        if stale is not None:
            log.warning("%s", stale)
        stubs = ", ".join(engine.stub_heads) or "none"
        log.info("api %s ready: bundle %s, stub heads %s, device %s", API_VERSION, engine.bundle.version, stubs, engine.device)
        try:
            yield
        finally:
            engine.close()

    app = FastAPI(
        title="facemeasure api",
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.lock = threading.Lock()
    app.state.max_pixels = max_pixels
    app.state.web_mounted = False
    app.add_middleware(RequestGuard)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error(request, exc.code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return _error(request, schemas.INVALID_REQUEST)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 400:  # the body does not parse; 400 is kept for images that do not decode
            return _error(request, schemas.INVALID_REQUEST)
        code = next((c for c, s in schemas.ERROR_STATUS.items() if s == exc.status_code), schemas.INVALID_REQUEST)
        return _error(request, code, status=exc.status_code, headers=getattr(exc, "headers", None))

    @app.get("/health", response_model=schemas.Health, responses={500: ERROR_RESPONSES[500]})
    async def health(request: Request) -> JSONResponse:
        engine: Engine = request.app.state.engine
        body = schemas.Health(
            status="ok",
            versions={"api": API_VERSION, **engine.versions()},
            stub_heads=engine.stub_heads,
            device=engine.device,
            languages=engine.languages,
        )
        return JSONResponse(body.model_dump(mode="json"))

    @app.post("/analyze", response_model=schemas.AnalyzeResult, responses=ERROR_RESPONSES)
    def analyze(
        request: Request,
        frontal: Annotated[UploadFile, File(description="frontal image, JPEG or PNG")],
        meta: Annotated[str, Form(description="JSON text of components.schemas.Meta")],
        profile: Annotated[UploadFile | None, File(description="profile image, JPEG or PNG; an empty file is no image")] = None,
    ) -> JSONResponse:
        m = parse_meta(meta)
        frontal_bytes, profile_bytes = read_upload(frontal), read_upload(profile)
        if not frontal_bytes:
            raise ApiError(schemas.INVALID_REQUEST)
        engine: Engine = request.app.state.engine
        max_px = request.app.state.max_pixels
        with request.app.state.lock:
            frontal_image = decode(frontal_bytes, max_px)
            profile_image = decode(profile_bytes, max_px) if profile_bytes else None
            del frontal_bytes, profile_bytes
            result = engine.analyze(
                frontal_image,
                profile_image,
                response.Meta(age=m.age, sex=m.sex, frontal_source=m.frontal_source, lang=m.lang),
                lab=m.lab,
            )
            del frontal_image, profile_image
        body = schemas.AnalyzeResult.model_validate(
            {
                "request_id": request.scope["state"][REQUEST_ID],
                "versions": {"api": API_VERSION, "model_bundle": engine.bundle.version},
                "lab": m.lab,
                **result,
            }
        )
        return JSONResponse(body.model_dump(mode="json"))

    def openapi() -> dict:
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            schema.setdefault("components", {}).setdefault("schemas", {})["Meta"] = schemas.Meta.model_json_schema()
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi

    if web.is_dir():  # after every endpoint, so / never hides them (web.md section 8.3)
        app.mount("/", StaticFiles(directory=web, html=True), name=WEB_MOUNT)
        app.state.web_mounted = True
    return app


app = create_app()
