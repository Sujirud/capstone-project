"""api/try_api.py: the multipart body it builds, its summary and its errors, without a running api."""

import io
import json
import urllib.error

import pytest

from api import try_api

JPEG_HEAD = b"\xff\xd8\xff\xe0" + b"\x00" * 20
PNG_HEAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


def test_multipart_body_parses_with_the_fields_and_generic_names():
    from email.parser import BytesParser
    from email.policy import HTTP

    body, content_type = try_api.multipart({"frontal": JPEG_HEAD, "profile": PNG_HEAD}, {"meta": '{"age": 22, "sex": "female"}'})
    msg = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    parts = {p.get_param("name", header="content-disposition"): p for p in msg.iter_parts()}
    assert set(parts) == {"meta", "frontal", "profile"}
    assert json.loads(parts["meta"].get_content()) == {"age": 22, "sex": "female"}
    assert parts["frontal"].get_content_type() == "image/jpeg" and parts["profile"].get_content_type() == "image/png"
    assert parts["frontal"].get_filename() == "frontal" and parts["frontal"].get_content() == JPEG_HEAD


def test_unknown_bytes_are_sent_as_they_are_so_the_api_can_answer_415():
    assert try_api.media_type(b"GIF89a") == "application/octet-stream"


def test_summary_of_the_api_md_example(api_example):
    text = try_api.summary(api_example, 1.234)
    assert text.isascii()
    lines = text.splitlines()
    assert lines[0].startswith(f"request {api_example['request_id']}  1.23 s")
    jaw = next(line for line in lines if line.startswith("jaw"))
    assert "assessed" in jaw and "0-1" in jaw and "83.6 %" in jaw and "(no_procedure_at_grade)" in jaw
    dorsum = next(line for line in lines if line.startswith("nose_dorsum"))
    assert "2+" in dorsum and "28.5 deg" in dorsum and "nose_filler,rhinoplasty_implant" in dorsum
    assert "retake: none" in lines


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def images(tmp_path):
    frontal, profile = tmp_path / "f.jpg", tmp_path / "p.png"
    frontal.write_bytes(JPEG_HEAD)
    profile.write_bytes(PNG_HEAD)
    return frontal, profile


def test_main_sends_the_meta_and_prints_the_summary(monkeypatch, capsys, images, api_example):
    sent = []

    def urlopen(request, timeout):
        sent.append(request)
        return FakeResponse(json.dumps(api_example).encode())

    monkeypatch.setattr(try_api.urllib.request, "urlopen", urlopen)
    frontal, profile = images
    assert try_api.main([str(frontal), str(profile), "--age", "22", "--sex", "female", "--lab", "--upload", "--port", "8123"]) == 0
    (req,) = sent
    assert req.full_url == "http://127.0.0.1:8123/analyze" and req.get_method() == "POST"
    assert b'"lab": true' in req.data and b'"frontal_source": "upload"' in req.data and b'"age": 22' in req.data
    assert "nose_dorsum" in capsys.readouterr().out


def test_main_json_prints_the_whole_result(monkeypatch, capsys, images, api_example):
    monkeypatch.setattr(try_api.urllib.request, "urlopen", lambda r, timeout: FakeResponse(json.dumps(api_example).encode()))
    monkeypatch.setattr(try_api.sys.stdout, "reconfigure", lambda **kw: None, raising=False)
    assert try_api.main([str(images[0]), "--age", "22", "--sex", "female", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == api_example


def test_main_reports_api_errors_and_a_missing_api(monkeypatch, capsys, images):
    def http_error(request, timeout):
        body = io.BytesIO(json.dumps({"error": "unsupported_media_type", "request_id": "abc"}).encode())
        raise urllib.error.HTTPError(request.full_url, 415, "Unsupported", {}, body)

    monkeypatch.setattr(try_api.urllib.request, "urlopen", http_error)
    assert try_api.main([str(images[0]), "--age", "22", "--sex", "male"]) == 1
    assert "HTTP 415: unsupported_media_type (request abc)" in capsys.readouterr().err

    def refused(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))

    monkeypatch.setattr(try_api.urllib.request, "urlopen", refused)
    assert try_api.main([str(images[0]), "--age", "22", "--sex", "male"]) == 1
    assert "uv run python -m api" in capsys.readouterr().err
