"""The api through FastAPI's TestClient with a fake engine (api.md sections 2-6).

The fake engine skips MediaPipe and DINOv2: measure() returns synthetic results from
conftest, and everything after it is real, so image_io decodes the uploads,
response.build runs the rules and texts, and schemas.py checks the result.
"""

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.formparsers import MultiPartParser

from api import API_VERSION, schemas
from api.engine import Engine, Measured, Parts
from api.main import (
    BUILD_COMMAND,
    LOGGER_NAME,
    create_app,
    web_build_warning,
    web_sources,
)
from api.try_api import multipart
from facemeasure import codes, paths, quality
from facemeasure.i18n import Texts
from facemeasure.profile import missing_profile
from facemeasure.response import Predictions
from make_stub_bundle import stub_bundle

PREDICTIONS = Predictions(jaw=0.8137, nose_base=1.0712, undereye_left=(0.1234, 0.0567), undereye_right=(0.4321, 0.0765))
LINE = re.compile(r"request_id=[0-9a-f-]{36} status=\d{3} duration_ms=\d+ error=([a-z_]+|-)")
DEFAULT_META = {"age": 30, "sex": "female"}


def image_bytes(fmt="JPEG", size=(64, 48), seed=0, **save) -> bytes:
    rgb = np.random.default_rng(seed).integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, fmt, **save)
    return buf.getvalue()


JPEG = image_bytes()
PNG = image_bytes("PNG", seed=1)


class FakeEngine(Engine):
    """Engine without models: measure() gives what the test asks for and records what it got."""

    def __init__(self, bundle, config, measured):
        super().__init__(Parts(config=config, texts=Texts(), bundle=bundle, warnings=()), models=None, backbone=None, device="cpu")
        self._measured = measured
        self.calls: list[dict] = []

    def measure(self, frontal, profile):
        self.calls.append({"frontal": frontal.shape, "profile": None if profile is None else profile.shape})
        return self._measured(frontal, profile)


@pytest.fixture
def measured(frontal_result, profile_result):
    """Factory of measure() functions: the example face, the example profile when one was sent."""

    def make(frontal=None, profile=None, predictions=PREDICTIONS, **frontal_kw):
        def measure(frontal_image, profile_image):
            f = frontal if frontal is not None else frontal_result(**frontal_kw)
            if profile_image is None:
                p = missing_profile()
            else:
                p = profile if profile is not None else profile_result()
            return Measured(frontal=f, profile=p, predictions=predictions)

        return measure

    return make


@pytest.fixture
def make_client(tmp_path, ready_bundle, measured):
    """Factory of started TestClients around a FakeEngine; returns (client, engine)."""
    started = []

    def make(bundle=None, measure=None, web_dist=None, max_pixels=None, max_bytes=None, sources=None):
        config = quality.load_config()
        if max_bytes is not None:
            config = replace(config, request=replace(config.request, max_bytes=max_bytes))
        engine = FakeEngine(bundle or ready_bundle, config, measure or measured())
        kw = {} if max_pixels is None else {"max_pixels": max_pixels}
        if sources is not None:
            kw["sources"] = sources
        app = create_app(lambda: engine, web_dist=web_dist or tmp_path / "no_web", **kw)
        client = TestClient(app)
        client.__enter__()
        started.append(client)
        return client, engine

    yield make
    for client in started:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()[0]


def post(client, frontal=JPEG, profile=PNG, meta=None, raw_meta=None, files=None, data=None):
    """POST /analyze with the usual parts; None leaves a part out."""
    if files is None:
        files = {}
        if frontal is not None:
            files["frontal"] = ("frontal.jpg", frontal, "image/jpeg")
        if profile is not None:
            files["profile"] = ("profile.png", profile, "image/png")
    if data is None:
        data = {}
        if raw_meta is not None:
            data["meta"] = raw_meta
        else:
            data["meta"] = json.dumps(DEFAULT_META if meta is None else meta)
    return client.post("/analyze", files=files or None, data=data)


def assert_error(res, status, code):
    assert res.status_code == status
    body = res.json()
    assert list(body) == ["error", "request_id"] and body["error"] == code
    assert uuid.UUID(body["request_id"]).version == 4
    assert res.headers["cache-control"] == "no-store"
    schemas.ErrorBody.model_validate(body)


# ---------------------------------------------------------------- health and the result


def test_health_has_every_field(make_client, ready_bundle):
    client, _ = make_client()
    res = client.get("/health")
    assert res.status_code == 200 and res.headers["cache-control"] == "no-store"
    body = res.json()
    assert list(body) == ["status", "versions", "stub_heads", "device", "languages"]
    assert body["status"] == "ok" and body["stub_heads"] == [] and body["device"] == "cpu" and body["languages"] == ["th", "en"]
    assert list(body["versions"]) == ["api", "model_bundle", "facemeasure", "mediapipe", "torch"]
    assert body["versions"]["api"] == API_VERSION and body["versions"]["model_bundle"] == ready_bundle.version
    schemas.Health.model_validate(body)

    stub_client, _ = make_client(bundle=stub_bundle())
    assert stub_client.get("/health").json()["stub_heads"] == ["jaw", "nose_base", "undereye"]


@pytest.mark.parametrize("lab", [False, True])
def test_analyze_gives_the_checked_result(make_client, ready_bundle, lab):
    client, engine = make_client()
    res = post(client, meta={**DEFAULT_META, "lab": lab})
    assert res.status_code == 200 and res.headers["cache-control"] == "no-store"
    assert res.headers["content-type"].startswith("application/json")
    body = res.json()
    schemas.AnalyzeResult.model_validate(body)
    assert list(body) == ["request_id", "versions", "lab", "retake", "quality", "regions", "disclaimer"]
    assert uuid.UUID(body["request_id"]).version == 4
    assert body["versions"] == {"api": API_VERSION, "model_bundle": ready_bundle.version}
    assert body["lab"] is lab
    assert [r["region"] for r in body["regions"]] == list(codes.REGIONS)
    assert all(r["status"] == "assessed" for r in body["regions"])
    jaw = next(r for r in body["regions"] if r["region"] == "jaw")
    assert jaw["measurements"][0]["value"] == 81.4  # the fake head's 0.8137, in percent with one digit
    # the uploads reach the engine decoded, at their own size
    assert engine.calls == [{"frontal": (48, 64, 3), "profile": (48, 64, 3)}]
    assert post(client).json()["request_id"] != body["request_id"]


def test_exif_orientation_is_applied_before_the_engine(make_client):
    client, engine = make_client()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees to show
    rotated = image_bytes(size=(64, 48), exif=exif.tobytes())
    assert post(client, frontal=rotated, profile=None).status_code == 200
    assert engine.calls[-1] == {"frontal": (64, 48, 3), "profile": None}


@pytest.mark.parametrize("send", ["empty_file", "left_out", "empty_text"])
def test_profile_is_optional_and_an_empty_file_is_none(make_client, send):
    client, engine = make_client()
    if send == "empty_file":
        res = post(client, profile=b"")
    elif send == "left_out":
        res = post(client, profile=None)
    else:
        res = post(client, files={"frontal": ("f.jpg", JPEG, "image/jpeg")}, data={"meta": json.dumps(DEFAULT_META), "profile": ""})
    assert res.status_code == 200
    assert engine.calls[-1]["profile"] is None
    body = res.json()
    regions = {r["region"]: r for r in body["regions"]}
    assert regions["chin"]["status_reasons"] == ["missing_input"] and regions["nose_dorsum"]["status_reasons"] == ["missing_input"]
    assert body["retake"][-1]["image"] == "profile" and body["retake"][-1]["code"] == "missing_input"
    assert body["quality"]["profile"] == {"face_found": None, "face_yaw_deg": None, "silhouette_found": None, "flipped": None,
                                          "failed": ["missing_input"]}


def test_the_example_of_api_md_passes_the_result_model(api_example):
    checked = schemas.AnalyzeResult.model_validate(api_example)
    assert checked.model_dump(mode="json")["regions"][1]["procedures"][0]["code"] == "nose_filler"


def test_the_result_model_refuses_unknown_fields_and_codes(api_example):
    import copy

    from pydantic import ValidationError

    for change in (
        lambda d: d.update(extra=1),
        lambda d: d["regions"][0].update(status="maybe"),
        lambda d: d["regions"][0]["status_reasons"].append("tired"),
        lambda d: d["quality"]["frontal"].pop("iod_px"),
        lambda d: d.update(request_id="not-a-uuid"),
    ):
        bad = copy.deepcopy(api_example)
        change(bad)
        with pytest.raises(ValidationError):
            schemas.AnalyzeResult.model_validate(bad)


def test_quality_keys_match_response_py():
    from facemeasure import response

    frontal = [k for k in schemas.FrontalQuality.model_fields if k != "failed"]
    profile = [k for k in schemas.ProfileQuality.model_fields if k != "failed"]
    assert frontal == list(response.FRONTAL_QUALITY_DIGITS) and profile == list(response.PROFILE_QUALITY_FIELDS)


# ---------------------------------------------------------------- 422


SECRET = "zz_secret_value_zz"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frontal": None},
        {"raw_meta": None, "data": {}},
        {"raw_meta": "{not json"},
        {"raw_meta": f'{{"age": 30, "sex": "female", "{SECRET}": 1}}'},
        {"meta": {"age": "31", "sex": "female"}},
        {"meta": {"age": 31.5, "sex": "female"}},
        {"meta": {"age": 31.0, "sex": "female"}},
        {"meta": {"age": True, "sex": "female"}},
        {"meta": {"age": -1, "sex": "female"}},
        {"meta": {"age": 121, "sex": "female"}},
        {"meta": {"sex": "female"}},
        {"meta": {"age": 30}},
        {"meta": {"age": 30, "sex": SECRET}},
        {"meta": {"age": 30, "sex": "female", "frontal_source": SECRET}},
        {"meta": {"age": 30, "sex": "female", "lab": "yes"}},
        {"meta": {"age": 30, "sex": "female", "lang": f"../{SECRET}"}},
        {"raw_meta": "[30, \"female\"]"},
        {"frontal": b""},
        {"files": {"frontal": ("f.jpg", JPEG, "image/jpeg")}, "data": {"meta": json.dumps(DEFAULT_META), "profile": SECRET}},
        {"files": {"meta": ("m.json", json.dumps(DEFAULT_META).encode(), "application/json"),
                   "frontal": ("f.jpg", JPEG, "image/jpeg")}, "data": {}},
    ],
)
def test_invalid_requests_get_422_without_what_was_sent(make_client, kwargs):
    client, engine = make_client()
    res = post(client, **kwargs)
    assert_error(res, 422, "invalid_request")
    assert SECRET not in res.text and "female" not in res.text and "detail" not in res.text
    assert engine.calls == []


def test_a_body_that_is_not_multipart_is_invalid(client):
    assert_error(client.post("/analyze", content=b"--x\r\nbroken", headers={"content-type": "multipart/form-data; boundary=x"}),
                 422, "invalid_request")
    assert_error(client.post("/analyze", json=DEFAULT_META), 422, "invalid_request")


# ---------------------------------------------------------------- 413, 415 and 400


def test_content_length_over_the_limit_gets_413(make_client):
    client, engine = make_client(max_bytes=4000)
    res = post(client, frontal=image_bytes("PNG", size=(64, 64), seed=3), profile=None)
    assert int(res.request.headers["content-length"]) > 4000
    assert_error(res, 413, "too_large")
    assert engine.calls == []
    assert post(client, frontal=image_bytes("PNG", size=(8, 8)), profile=None).status_code == 200


def test_chunked_body_over_the_limit_gets_413(make_client):
    client, engine = make_client(max_bytes=4000)
    body, content_type = multipart({"frontal": image_bytes("PNG", size=(64, 64), seed=3)}, {"meta": json.dumps(DEFAULT_META)})
    res = client.post("/analyze", content=iter([body]), headers={"content-type": content_type})
    assert "content-length" not in res.request.headers
    assert_error(res, 413, "too_large")
    assert engine.calls == []


def _asgi_scope(app, headers):
    return {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": "/analyze", "raw_path": b"/analyze", "root_path": "", "query_string": b"", "headers": headers,
        "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 8000), "app": app,
    }


def test_the_guard_refuses_a_large_content_length_without_reading_the_body(make_client):
    client, engine = make_client(max_bytes=1000)
    sent = []

    async def receive():
        raise AssertionError("the body was read")

    async def send(message):
        sent.append(message)

    headers = [(b"content-length", b"5000"), (b"content-type", b"multipart/form-data; boundary=x")]
    asyncio.run(client.app(_asgi_scope(client.app, headers), receive, send))
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 413
    assert (b"cache-control", b"no-store") in sent[0]["headers"]
    assert json.loads(sent[1]["body"])["error"] == "too_large"
    assert engine.calls == []


def test_the_guard_stops_a_chunked_body_while_it_is_read(make_client):
    client, engine = make_client(max_bytes=1000)
    sent, reads = [], []

    head = b'--x\r\nContent-Disposition: form-data; name="frontal"; filename="f"\r\nContent-Type: image/png\r\n\r\n'
    body = head + PNG * 10

    async def receive():
        reads.append(1)
        if len(reads) > 10:
            raise AssertionError("the body kept being read after the limit")
        chunk = body[(len(reads) - 1) * 600 : len(reads) * 600]
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    headers = [(b"content-type", b"multipart/form-data; boundary=x"), (b"transfer-encoding", b"chunked")]
    asyncio.run(client.app(_asgi_scope(client.app, headers), receive, send))
    assert len(reads) == 2  # 600, then 1200 > 1000
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1 and starts[0]["status"] == 413
    assert engine.calls == []


def test_non_images_get_415(client):
    gif = image_bytes("GIF")
    assert gif.startswith(b"GIF8")
    assert_error(post(client, frontal=gif), 415, "unsupported_media_type")
    assert_error(post(client, frontal=b"just text, not an image"), 415, "unsupported_media_type")
    assert_error(post(client, profile=gif), 415, "unsupported_media_type")


def test_a_cut_jpeg_gets_400(client):
    cut = JPEG[: len(JPEG) // 2]
    assert_error(post(client, frontal=cut), 400, "decode_failed")
    assert_error(post(client, profile=cut), 400, "decode_failed")


def test_too_many_pixels_get_413(make_client):
    client, engine = make_client(max_pixels=1000)
    assert_error(post(client, frontal=image_bytes(size=(64, 48))), 413, "too_large")
    assert engine.calls == []


# ---------------------------------------------------------------- language and stub heads


def test_a_language_without_a_file_gets_thai(client):
    thai = Texts().text("disclaimer", "th")
    for lang in ("xx", "fr", "th"):
        body = post(client, meta={**DEFAULT_META, "lang": lang}).json()
        assert body["disclaimer"] == thai
        assert body["regions"][0]["measurements"][0]["reference"].startswith("จุดตัด")


THAI = re.compile("[\u0e00-\u0e7f]")


@pytest.mark.parametrize("source", ["camera", "upload"])
@pytest.mark.parametrize("sex", ["female", "male"])
@pytest.mark.parametrize("age", [17, 21, 30])
def test_english_results_have_no_thai(make_client, measured, profile_result, age, sex, source):
    """shared/i18n/en.json covers every text the api writes: procedures, notes, withheld and criteria."""
    client, _ = make_client(measure=measured(predictions=TOP, profile=profile_result(legan=26.0, nasofacial=25.0)))
    meta = {"age": age, "sex": sex, "frontal_source": source, "lang": "en"}
    for profile in (PNG, None):
        body = post(client, profile=profile, meta=meta).json()
        texts = json.dumps(body, ensure_ascii=False)
        assert not THAI.search(texts), THAI.findall(texts)[:5]
        assert body["disclaimer"] == Texts().text("disclaimer", "en")


def test_stub_heads_give_model_not_ready_by_mode(make_client):
    client, _ = make_client(bundle=stub_bundle())
    body = post(client, profile=None).json()
    regions = {r["region"]: r for r in body["regions"]}
    for name in codes.FRONTAL_REGIONS:
        assert regions[name]["status"] == "not_assessable" and regions[name]["status_reasons"] == ["model_not_ready"]
    assert regions["chin"]["status_reasons"] == ["missing_input"]
    assert [(i["image"], i["code"]) for i in body["retake"]] == [("profile", "missing_input")]
    assert all(name not in codes.FRONTAL_REGIONS for i in body["retake"] for name in i["regions"])

    body = post(client, meta={**DEFAULT_META, "lab": True}).json()
    assert body["retake"] == [] and body["lab"] is True
    for r in body["regions"]:
        assert r["status"] == "assessed"
        assert r["warnings"] == (["model_not_ready"] if r["region"] in codes.FRONTAL_REGIONS else [])


# ---------------------------------------------------------------- no files, no values in the log


def test_a_5_mb_upload_never_touches_the_disk(make_client, measured, tmp_path, monkeypatch):
    folder = tmp_path / "tmp"
    folder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(folder))
    monkeypatch.setattr(MultiPartParser, "spool_max_size", 1024 * 1024)  # the default; the api must raise it
    rollovers = []
    original = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", lambda self: (rollovers.append(1), original(self))[1])
    during = []
    inner = measured()

    def measure(frontal, profile):
        during.append(sorted(p.name for p in folder.iterdir()))
        return inner(frontal, profile)

    client, _ = make_client(measure=measure)
    big = image_bytes("PNG", size=(1300, 1300), seed=5)
    assert len(big) > 5_000_000
    assert MultiPartParser.spool_max_size >= quality.load_config().request.max_bytes
    assert post(client, frontal=big, profile=None).status_code == 200
    assert during == [[]] and rollovers == []
    assert list(folder.iterdir()) == []


class Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def lines():
    handler = Lines()
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    yield handler.lines
    logger.removeHandler(handler)


def test_one_log_line_per_request_without_meta_or_values(make_client, lines):
    client, _ = make_client()
    lines.clear()
    ok = post(client, meta={"age": 37, "sex": "male", "frontal_source": "upload"})
    bad = post(client, meta={"age": 37, "sex": "nobody"})
    big = post(client, frontal=image_bytes("GIF"))
    client.get("/health")
    assert len(lines) == 4
    for line in lines:
        assert LINE.fullmatch(line), line
    assert ok.json()["request_id"] in lines[0] and "status=200" in lines[0] and lines[0].endswith("error=-")
    assert bad.json()["request_id"] in lines[1] and "status=422" in lines[1] and lines[1].endswith("error=invalid_request")
    assert big.json()["request_id"] in lines[2] and "status=415" in lines[2] and lines[2].endswith("error=unsupported_media_type")
    # LINE already allows nothing else; spelled out for the values the requests and the fake engine had
    text = "\n".join(lines)
    for value in ("male", "upload", "nobody", "81.4", "0.8137", "1.071", "age", "sex"):
        assert value not in text


def test_an_unexpected_error_is_500_with_nothing_in_the_log(make_client, lines):
    def broken(frontal, profile):
        raise RuntimeError(f"broke at jaw 0.8137 for {SECRET}")

    client, _ = make_client(measure=broken)
    lines.clear()
    res = post(client)
    assert_error(res, 500, "internal_error")
    assert SECRET not in res.text
    assert len(lines) == 1 and LINE.fullmatch(lines[0]) and "status=500" in lines[0]
    assert lines[0].endswith("error=internal_error") and SECRET not in lines[0] and "0.8137" not in lines[0]


def test_a_result_that_does_not_fit_the_model_is_500(make_client, monkeypatch):
    client, engine = make_client()
    real = engine.analyze
    monkeypatch.setattr(engine, "analyze", lambda *a, **kw: {**real(*a, **kw), "surprise": 1})
    assert_error(post(client), 500, "internal_error")


def test_requests_run_one_at_a_time(make_client, measured):
    active, peak = [0], [0]
    lock = threading.Lock()
    inner = measured()

    def slow(frontal, profile):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return inner(frontal, profile)

    client, engine = make_client(measure=slow)
    results = []
    threads = [threading.Thread(target=lambda: results.append(post(client).status_code)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 4 and len(engine.calls) == 4 and peak[0] == 1


# ---------------------------------------------------------------- docs, web page and other paths


def test_openapi_stays_and_the_cdn_pages_are_off(client):
    spec = client.get("/openapi.json")
    assert spec.status_code == 200 and spec.headers["cache-control"] == "no-store"
    schemas_ = spec.json()["components"]["schemas"]
    assert {"Meta", "AnalyzeResult", "ErrorBody", "Health"} <= set(schemas_)
    assert set(schemas_["Meta"]["properties"]) == {"age", "sex", "frontal_source", "lab", "lang"}
    assert client.get("/docs").status_code == 404 and client.get("/redoc").status_code == 404


def test_without_a_web_build_only_the_api_answers(client):
    assert_error(client.get("/"), 404, "invalid_request")
    assert_error(client.get("/analyze"), 405, "invalid_request")


def test_the_web_build_is_served_at_the_root(make_client, tmp_path):
    web = tmp_path / "dist"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>web</title><p>marble</p>", encoding="utf-8")
    (web / "assets").mkdir()
    (web / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    client, _ = make_client(web_dist=web)
    page = client.get("/")
    assert page.status_code == 200 and "marble" in page.text and page.headers["cache-control"] == "no-store"
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/health").json()["status"] == "ok"
    assert post(client).status_code == 200
    assert_error(client.get("/missing.png"), 404, "invalid_request")


def _touch(path, text, stamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (stamp, stamp))
    return path


def test_web_build_warning_when_missing_or_older_than_a_source(tmp_path):
    src, page = tmp_path / "src", tmp_path / "index.html"
    _touch(src / "main.ts", "1", 100)
    _touch(page, "<p>", 100)
    dist = tmp_path / "dist"
    missing = web_build_warning(dist, [src, page])
    assert "not built" in missing and BUILD_COMMAND in missing
    _touch(dist / "index.html", "<p>", 200)
    assert web_build_warning(dist, [src, page, tmp_path / "absent"]) is None
    newer = _touch(src / "screens" / "home.ts", "2", 300)  # a nested file changed after the build
    stale = web_build_warning(dist, [src, page])
    assert str(newer) in stale and BUILD_COMMAND in stale
    _touch(dist / "index.html", "<p>", 400)
    assert web_build_warning(dist, [src, page]) is None
    _touch(page, "<p>", 500)  # a single file as a source
    assert str(page) in web_build_warning(dist, [src, page])


def test_web_build_sources_are_the_files_the_build_includes():
    names = {p.relative_to(paths.project_root()).as_posix() for p in web_sources()}
    assert names == {"web/src", "web/index.html", "shared/quality.json", "shared/i18n"}
    assert all(p.exists() for p in web_sources())


def test_the_api_warns_at_start_about_a_stale_web_build(make_client, tmp_path, lines):
    dist = tmp_path / "dist"
    _touch(dist / "index.html", "<p>", 100)
    newer = _touch(tmp_path / "src" / "main.ts", "1", 200)
    lines.clear()
    client, _ = make_client(web_dist=dist, sources=[tmp_path / "src"])
    assert any(str(newer) in line and BUILD_COMMAND in line for line in lines)
    assert client.get("/").status_code == 200  # still served, with the warning
    lines.clear()
    _touch(dist / "index.html", "<p>", 300)
    make_client(web_dist=dist, sources=[tmp_path / "src"])
    assert not any("web build" in line or "not built" in line for line in lines)


# ---------------------------------------------------------------- the rules through the api

# Every region at its top grade: the jaw and the alar base over the last cutoff for both
# sexes, the chin at grade 3, both under-eye sides at 2+ and the dorsum below the smallest
# cutoff of both sexes.
TOP = Predictions(jaw=0.90, nose_base=1.45, undereye_left=(0.95, 0.90), undereye_right=(0.95, 0.90))
TOP_PROCEDURES = {
    "jaw": ["botox_masseter", "jaw_reduction"],
    "chin": ["genioplasty"],
    "undereye_left": ["undereye_filler"],
    "undereye_right": ["undereye_filler"],
    "nose_base": ["alar_reduction"],
    "nose_dorsum": ["nose_filler", "rhinoplasty_implant"],
}
TOP_NOTES = {
    "jaw": ["jaw_cause_unknown"],
    "chin": ["chin_check_mandible"],
    "undereye_left": ["undereye_scope"],
    "undereye_right": ["undereye_scope"],
    "nose_base": ["nose_base_indications"],
    "nose_dorsum": ["nose_filler_vision_risk"],
}


def expected_withheld(region: str, age: int, sex: str, source: str) -> list[str]:
    """Written out from model.md sections 1.3 and 4.3 rather than from rules.py."""
    out = []
    if region == "jaw" and sex == "male":
        out.append("male_jaw_no_criterion")
    if region == "nose_base" and source == "upload":
        out.append("distance_unknown")
    if age < 18:
        out.append("age_under_18")
    elif region.startswith("undereye") and age <= 21:
        out.append("age_21_or_under")
    return out


@pytest.mark.parametrize("source", ["camera", "upload"])
@pytest.mark.parametrize("sex", ["female", "male"])
@pytest.mark.parametrize("age", [17, 18, 21, 22])
def test_rules_through_the_api(make_client, measured, profile_result, age, sex, source):
    client, _ = make_client(measure=measured(predictions=TOP, profile=profile_result(legan=26.0, nasofacial=25.0)))
    body = post(client, meta={"age": age, "sex": sex, "frontal_source": source}).json()
    for r in body["regions"]:
        name = r["region"]
        assert r["status"] == "assessed"
        withheld = expected_withheld(name, age, sex, source)
        assert r["procedures_withheld"] == withheld, name
        assert [p["code"] for p in r["procedures"]] == ([] if withheld else TOP_PROCEDURES[name]), name
        notes = [n["code"] for n in r["notes"]]
        if withheld:
            assert notes == (["undereye_scope"] if name.startswith("undereye") else []), name
        else:
            assert notes == TOP_NOTES[name], name
        no_grade = (name == "jaw" and sex == "male") or (name == "nose_base" and source == "upload")
        assert (r["decision"] is None) == no_grade, name
        assert r["measurements"] or name.startswith("undereye")
