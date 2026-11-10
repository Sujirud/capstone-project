"""Language files (shared/i18n/) and i18n.py (web.md section 8.4)."""

import json
import re

import pytest

from facemeasure import codes, criteria, i18n, paths
from facemeasure.i18n import MissingText, TextError, Texts

TH_PATH = i18n.folder() / "th.json"
THAI = re.compile("[฀-๿]")


@pytest.fixture(scope="module")
def th() -> dict[str, str]:
    with open(TH_PATH, encoding="utf-8") as f:
        return json.load(f)


def required_keys() -> set[str]:
    keys = {"disclaimer", "sex.female", "sex.male", "language.name", "measure.method.model", "quality.too_close"}
    for region in (*codes.REGIONS, "undereye"):
        keys |= {f"region.{region}.name", f"region.{region}.subtitle"}
    for region in ("jaw", "chin", "undereye", "nose_base", "nose_dorsum"):
        keys |= {f"region.{region}.what", f"region.{region}.how"}
    keys |= {f"level.{x}" for x in ("clear", "borderline", "values_only", "not_assessable", "upload")}
    keys |= {f"grade.default.{g}" for g in criteria.GRADES} | {f"grade.chin.{g}" for g in criteria.CHIN_GRADES}
    keys |= {"grade.between", "grade.between_wide"}
    for m in codes.MEASURES:
        keys |= {f"measure.{m}.name", f"measure.{m}.subtitle"}
    keys |= {"reference.facial_taperness.female", "reference.facial_taperness.male"}
    keys |= {f"reference.{m}" for m in codes.MEASURES if m != "facial_taperness"}
    for p in codes.PROCEDURES:
        keys |= {f"procedure.{p}.name", f"procedure.{p}.when"}
    keys |= {f"note.{n}" for n in codes.NOTES}
    keys |= {f"withheld.{w}" for w in codes.WITHHELD}
    keys |= {f"quality.{q}" for q in codes.STATUS_CODES}
    return keys


def test_every_code_has_its_text(th):
    missing = sorted(required_keys() - set(th))
    assert not missing


def test_file_is_flat_text_written_the_same_on_every_machine(th):
    raw = TH_PATH.read_bytes()
    assert b"\r" not in raw
    assert raw.decode("utf-8") == json.dumps(th, ensure_ascii=False, indent=2) + "\n"
    assert all(isinstance(v, str) and v for v in th.values())


def test_placeholders_of_the_templates(th):
    assert i18n.placeholders(th["reference.facial_taperness.female"]) == {"cut", "band"}
    assert i18n.placeholders(th["reference.legan_angle"]) == {"ideal", "low", "high", "step"}
    assert i18n.placeholders(th["reference.alar_intercanthal_ratio"]) == {"sex", "low", "high", "g2"}
    assert i18n.placeholders(th["reference.nasofacial_angle"]) == {"sex", "low", "high", "g2"}
    assert i18n.placeholders(th["grade.between"]) == {"low", "high"}


def test_neutral_wording(th):
    # model.md section 1.3 and web.md section 1.2
    for key, text in th.items():
        for word in ("ผิดปกติ", "ไม่สวย"):
            assert word not in text, key


def test_age_texts_match_the_criteria(th):
    assert str(criteria.MIN_PROCEDURE_AGE) in th["withheld.age_under_18"]
    assert str(criteria.UNDEREYE_WITHHELD_MAX_AGE) in th["withheld.age_21_or_under"]


def doc_table(doc: str, heading: str) -> dict[str, str]:
    """`code` -> second column of the first table under a heading of docs/<doc>."""
    text = (paths.project_root() / "docs" / doc).read_text(encoding="utf-8")
    section = text.split(f"\n{heading}", 1)[1].split("\n#", 1)[0]
    return dict(re.findall(r"^\| `([a-z0-9_]+)` \| ([^|]+?) \|", section, flags=re.MULTILINE))


def test_procedure_names_are_those_of_api_md(th):
    names = doc_table("api.md", "**รหัสหัตถการและข้อความเพิ่มเติม**")
    for p in codes.PROCEDURES:
        assert th[f"procedure.{p}.name"] == names[p]


def test_withheld_and_quality_texts_are_those_of_web_md(th):
    withheld = doc_table("web.md", "### 5.2")
    assert {w: th[f"withheld.{w}"] for w in codes.WITHHELD} == {w: withheld[w] for w in codes.WITHHELD}
    quality = doc_table("web.md", "### 5.4")
    for code in codes.STATUS_CODES:
        assert quality[code].startswith(th[f"quality.{code}"]), code  # frontal_pose adds the direction in the web


def test_no_thai_in_the_package_code():
    found = [p.name for p in sorted((paths.project_root() / "src" / "facemeasure").rglob("*.py")) if THAI.search(p.read_text(encoding="utf-8"))]
    assert found == []


# ---------------------------------------------------------------- loading and fallback


def write(folder, lang, data):
    with open(folder / f"{lang}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False)


def test_project_files_load():
    texts = Texts()
    assert texts.codes[0] == "th"
    assert texts.text("sex.female") == "หญิง"
    assert texts.text("grade.between", low="0", high="1") == "อยู่ระหว่างระดับ 0 กับ 1"


def test_project_english_has_every_key_in_english():
    texts = Texts()
    th, en = texts.languages["th"], texts.languages["en"]
    assert list(en) == list(th)
    # the label of the language picker keeps the Thai word, so a Thai reader finds the way back
    assert [k for k, t in en.items() if k != "ui.lang.label" and THAI.search(t)] == []
    assert texts.text("grade.between", "en", low="0", high="1") == "Between grade 0 and 1"


def test_missing_keys_fall_back_to_thai(tmp_path):
    write(tmp_path, "th", {"a": "ก {x}", "b": "ข"})
    write(tmp_path, "en", {"a": "A {x}"})
    texts = Texts(tmp_path)
    assert texts.codes == ("th", "en")
    assert texts.text("a", "en", x=1) == "A 1"
    assert texts.text("b", "en") == "ข"
    assert texts.text("a", "fr", x=2) == "ก 2"  # no file: Thai
    assert texts.resolve("fr") == "th" and texts.resolve("en") == "en"


def test_key_missing_from_thai_is_an_error(tmp_path):
    write(tmp_path, "th", {"a": "ก"})
    with pytest.raises(MissingText):
        Texts(tmp_path).text("nope", "th")


def test_missing_value_is_an_error(tmp_path):
    write(tmp_path, "th", {"a": "ก {x}"})
    with pytest.raises(TextError, match="x"):
        Texts(tmp_path).text("a")


@pytest.mark.parametrize(
    "other, match",
    [
        ({"a": "A {y}"}, "placeholders"),
        ({"a": "A"}, "placeholders"),
        ({"z": "Z"}, "lacks"),
        ({"a": 3}, "text"),
    ],
)
def test_bad_language_files(tmp_path, other, match):
    write(tmp_path, "th", {"a": "ก {x}"})
    write(tmp_path, "en", other)
    with pytest.raises(TextError, match=match):
        Texts(tmp_path)


def test_thai_is_required(tmp_path):
    write(tmp_path, "en", {"a": "A"})
    with pytest.raises(TextError, match="th.json"):
        Texts(tmp_path)


def test_file_name_must_be_a_language_code(tmp_path):
    write(tmp_path, "th", {"a": "ก"})
    write(tmp_path, "Thai copy", {"a": "ก"})
    with pytest.raises(TextError, match="language code"):
        Texts(tmp_path)
