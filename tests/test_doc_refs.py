"""Every reference to a section of docs/ points at a heading that exists.

References look like "api.md หัวข้อ 6", "model.md หัวข้อ 4.1 และ 4.2", "api.md หัวข้อ 6–8"
in Thai text, or "web.md section 4.1" in code comments. Inside docs/<name>.md, a bare
"หัวข้อ 3.2" means a section of the same file. The reference list at the end of each doc
is skipped, because its "หัวข้อ" are sections of the papers.
"""

import re
from pathlib import Path

import pytest

from facemeasure import paths

DOCS = ("model", "api", "web")
NUM = r"\d+(?:\.\d+)*"
LIST = rf"{NUM}(?:\s*(?:,|และ|and|–|-)\s*{NUM})*"
CROSS = re.compile(rf"\b({'|'.join(DOCS)})\.md,? (?:หัวข้อ|sections?) ({LIST})")
APPENDIX = re.compile(rf"\b({'|'.join(DOCS)})\.md (ภาคผนวก \S+?)[\s.,)]")
SAME = re.compile(rf"หัวข้อ ({LIST})")
HEADING = re.compile(r"^#{2,4} (\d+(?:\.\d+)*)\.? ", re.MULTILINE)
APPENDIX_HEADING = re.compile(r"^## (ภาคผนวก \S+?)\.", re.MULTILINE)
SUFFIXES = {".md", ".py", ".ts", ".toml", ".json", ".html", ".css"}
SKIP_PARTS = {".venv", "node_modules", "dist", "data", "assets", "bundle", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def numbers(listed: str) -> list[str]:
    """'3.1, 3.2 และ 3.4' -> 3.1 3.2 3.4; '6–8' -> 6 7 8."""
    out = []
    for part in re.split(r"\s*(?:,|และ|and)\s*", listed):
        ends = re.split(r"\s*[–-]\s*", part)
        if len(ends) == 2 and all(e.isdigit() for e in ends):
            out += [str(n) for n in range(int(ends[0]), int(ends[1]) + 1)]
        else:
            out += ends
    return out


def headings(text: str) -> set[str]:
    return set(HEADING.findall(text)) | set(APPENDIX_HEADING.findall(text))


def project_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix in SUFFIXES and not SKIP_PARTS & set(p.relative_to(root).parts)
    )


def broken_cross_refs(text: str, known: dict[str, set[str]]) -> list[str]:
    found = [f"{m.group(1)}.md {n}" for m in CROSS.finditer(text) for n in numbers(m.group(2)) if n not in known[m.group(1)]]
    return found + [f"{m.group(1)}.md {m.group(2)}" for m in APPENDIX.finditer(text) if m.group(2) not in known[m.group(1)]]


def broken_same_file_refs(text: str, own: set[str]) -> list[str]:
    found = []
    for line in text.splitlines():
        if line.startswith("- \\["):  # the reference list
            continue
        for m in SAME.finditer(line):
            if re.search(rf"(?:{'|'.join(DOCS)})\.md,?\s*$", line[: m.start()]):
                continue  # a section of another file, checked by broken_cross_refs
            found += [n for n in numbers(m.group(1)) if n not in own]
    return found


@pytest.fixture(scope="module")
def known():
    docs = paths.project_root() / "docs"
    return {name: headings((docs / f"{name}.md").read_text(encoding="utf-8")) for name in DOCS}


def test_every_reference_across_files_has_its_heading(known):
    root = paths.project_root()
    broken = {}
    for f in project_files(root):
        if f.resolve() == Path(__file__).resolve():
            continue  # its synthetic text holds broken references on purpose
        bad = broken_cross_refs(f.read_text(encoding="utf-8"), known)
        if bad:
            broken[f.relative_to(root).as_posix()] = bad
    assert broken == {}


@pytest.mark.parametrize("name", DOCS)
def test_every_reference_inside_a_doc_has_its_heading(known, name):
    text = (paths.project_root() / "docs" / f"{name}.md").read_text(encoding="utf-8")
    assert broken_same_file_refs(text, known[name]) == []


def test_the_parser_on_synthetic_text():
    known = {"model": {"3", "3.1", "3.2", "ภาคผนวก ก"}, "api": {"6", "7", "8"}, "web": {"4.1"}}
    assert numbers("3.1, 3.2 และ 3.4") == ["3.1", "3.2", "3.4"] and numbers("6–8") == ["6", "7", "8"]
    text = "ดู model.md หัวข้อ 3.1 และ 3.2, api.md หัวข้อ 6–8 และ web.md section 4.1 กับ model.md ภาคผนวก ก."
    assert broken_cross_refs(text, known) == []
    bad = "model.md หัวข้อ 3.1 และ 3.9 กับ api.md หัวข้อ 7–9 และ web.md sections 4.1 and 5 และ model.md ภาคผนวก ข."
    assert broken_cross_refs(bad, known) == ["model.md 3.9", "api.md 9", "web.md 5", "model.md ภาคผนวก ข"]
    doc = "ตามหัวข้อ 3.1 และ api.md หัวข้อ 9\n- \\[11\\] โปรโตคอล หัวข้อ 7.2\nดูหัวข้อ 4"
    assert broken_same_file_refs(doc, {"3.1"}) == ["4"]
