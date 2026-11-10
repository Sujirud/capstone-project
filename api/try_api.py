"""Send a frontal image, and a profile image if given, to the running api and print a summary per region.

Start the api first (uv run python -m api), then from the project root:

    uv run python api/try_api.py frontal.jpg profile.jpg --age 22 --sex female
    uv run python api/try_api.py frontal.jpg --age 30 --sex male --lab --upload --lang th --json

For each region it prints the status, the grade (a range when borderline), the value,
the codes in status_reasons or warnings, and the procedure codes or why there are none.
--json prints the whole result instead. Neither the images nor the result are saved.
Only the Python standard library is used, so it runs anywhere uv does and stands in for
curl on Windows. The output is ASCII except with --json, so any console shows it.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HOST = "127.0.0.1"
DEFAULT_PORT = 8000
TIMEOUT_S = 120  # the web gives up at about 60 seconds; a first request can be slower
MEDIA_TYPES = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG\r\n\x1a\n": "image/png"}
COLUMNS = ("region", "status", "grade", "value", "codes", "procedures")
WIDTHS = (15, 15, 10, 20, 34, 0)


def media_type(data: bytes) -> str:
    """From the first bytes, as the api judges it; the api answers 415 for anything else."""
    return next((t for magic, t in MEDIA_TYPES.items() if data.startswith(magic)), "application/octet-stream")


def multipart(files: dict[str, bytes], fields: dict[str, str]) -> tuple[bytes, str]:
    """A multipart/form-data body and its Content-Type. File names are generic, so none leaks."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
            f"Content-Type: application/json; charset=utf-8\r\n\r\n".encode("ascii")
            + value.encode("utf-8")
            + b"\r\n"
        )
    for name, data in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}"\r\n'
            f"Content-Type: {media_type(data)}\r\n\r\n".encode("ascii")
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def grade_text(decision: dict | None) -> str:
    if decision is None:
        return "-"
    if decision["grade"] is not None:
        return decision["grade"]
    low, high = decision["grade_range"]
    return f"{low}-{high}"


def value_text(region: dict) -> str:
    if region["measurements"]:
        m = region["measurements"][0]
        return f"{m['value']:g} {m['unit']}"
    p = region["probabilities"]
    if p is not None:
        return f"P>=1 {p['at_least']['1']:.2f} >=2 {p['at_least']['2+']:.2f}"
    return "-"


def codes_text(region: dict) -> str:
    reasons = region["status_reasons"]
    warnings = [f"warn:{c}" for c in region["warnings"]]
    return ",".join(reasons + warnings) or "-"


def procedures_text(region: dict) -> str:
    if region["procedures"]:
        return ",".join(p["code"] for p in region["procedures"])
    if region["procedures_withheld"]:
        return "(" + ",".join(region["procedures_withheld"]) + ")"
    return "-"


def row(cells) -> str:
    return "".join(str(c).ljust(w) if w else str(c) for c, w in zip(cells, WIDTHS, strict=True)).rstrip()


def summary(result: dict, seconds: float) -> str:
    v = result["versions"]
    head = f"request {result['request_id']}  {seconds:.2f} s  api {v['api']}  bundle {v['model_bundle']}  lab {str(result['lab']).lower()}"
    lines = [
        head,
        "",
        row(COLUMNS),
    ]
    for r in result["regions"]:
        lines.append(row((r["region"], r["status"], grade_text(r["decision"]), value_text(r), codes_text(r), procedures_text(r))))
    lines.append("")
    if result["retake"]:
        for item in result["retake"]:
            lines.append(f"retake {item['image']}: {item['code']} ({', '.join(item['regions'])})")
    else:
        lines.append("retake: none")
    failed = {k: v["failed"] for k, v in result["quality"].items()}
    lines.append(f"quality failed: frontal {failed['frontal'] or '-'}, profile {failed['profile'] or '-'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send images to the running api and print a summary per region")
    parser.add_argument("frontal", type=Path, help="frontal image, JPEG or PNG")
    parser.add_argument("profile", type=Path, nargs="?", help="profile image, JPEG or PNG (optional)")
    parser.add_argument("--age", type=int, required=True, help="age in years, 0-120")
    parser.add_argument("--sex", choices=("female", "male"), required=True)
    parser.add_argument("--upload", action="store_true", help="the frontal image was picked from the computer (frontal_source upload)")
    parser.add_argument("--lab", action="store_true", help="lab mode: measure even when the images fail the checks")
    parser.add_argument("--lang", default="th", help="language of the texts (default th)")
    parser.add_argument("--json", action="store_true", help="print the whole JSON result")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port of the api (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)

    files = {"frontal": args.frontal.read_bytes()}
    if args.profile is not None:
        files["profile"] = args.profile.read_bytes()
    meta = {"age": args.age, "sex": args.sex, "frontal_source": "upload" if args.upload else "camera", "lab": args.lab, "lang": args.lang}
    body, content_type = multipart(files, {"meta": json.dumps(meta)})
    url = f"http://{HOST}:{args.port}/analyze"
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as res:
            result = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
        except ValueError:
            err = {}
        print(f"HTTP {e.code}: {err.get('error', '?')} (request {err.get('request_id', '?')})", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        print(f"Cannot reach the api at {url}: {reason}. Start it first: uv run python -m api", file=sys.stderr)
        return 1
    seconds = time.perf_counter() - start
    if args.json:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(summary(result, seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
