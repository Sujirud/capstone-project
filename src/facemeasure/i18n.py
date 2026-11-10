"""Texts of the results from the language files shared/i18n/<code>.json (web.md section 8.4).

Each file is one flat JSON object: dotted keys such as "procedure.chin_filler.name"
mapped to texts, which may hold placeholders like {low}. Thai (th.json) is the main
language and must have every key. Another language may leave keys out, and those fall
back to Thai; a key missing from Thai as well is a bug and raises MissingText. Adding a
language is adding a file, with no code change; its placeholders must match Thai's.
"""

import json
import re
from pathlib import Path

from facemeasure import paths

DEFAULT_LANG = "th"
FOLDER_NAME = "i18n"
PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
LANG_CODE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


class TextError(ValueError):
    """A language file is not a flat object of texts, or its placeholders differ from Thai."""


class MissingText(KeyError):
    """Thai has no text for this key: a bug in the code or in th.json."""


def placeholders(text: str) -> set[str]:
    return set(PLACEHOLDER.findall(text))


def folder() -> Path:
    return paths.path("shared") / FOLDER_NAME


def read_language(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TextError(f"{path.name} must hold one JSON object")
    for key, value in data.items():
        if not isinstance(value, str):
            raise TextError(f"{path.name}: {key} must be a text, got {type(value).__name__}")
    return data


class Texts:
    """Every language file of one folder, loaded once."""

    def __init__(self, source: Path | None = None):
        source = Path(source) if source is not None else folder()
        self.languages: dict[str, dict[str, str]] = {}
        for path in sorted(source.glob("*.json")):
            if not LANG_CODE.match(path.stem):
                raise TextError(f"{path.name}: the file name must be a language code such as th.json")
            self.languages[path.stem] = read_language(path)
        if DEFAULT_LANG not in self.languages:
            raise TextError(f"no {DEFAULT_LANG}.json in {source}")
        main = self.languages[DEFAULT_LANG]
        for lang, texts in self.languages.items():
            if lang == DEFAULT_LANG:
                continue
            extra = sorted(set(texts) - set(main))
            if extra:
                raise TextError(f"{lang}.json has keys that {DEFAULT_LANG}.json lacks: {', '.join(extra)}")
            for key, text in texts.items():
                if placeholders(text) != placeholders(main[key]):
                    raise TextError(
                        f"{lang}.json: {key} has placeholders {sorted(placeholders(text))}, "
                        f"{DEFAULT_LANG}.json has {sorted(placeholders(main[key]))}"
                    )

    @property
    def codes(self) -> tuple[str, ...]:
        """Language codes that have a file, Thai first."""
        return (DEFAULT_LANG, *sorted(k for k in self.languages if k != DEFAULT_LANG))

    def resolve(self, lang: str | None) -> str:
        """lang if it has a file, else Thai."""
        return lang if lang in self.languages else DEFAULT_LANG

    def raw(self, key: str, lang: str | None = DEFAULT_LANG) -> str:
        """The text of key in lang, or in Thai when lang has no such key."""
        texts = self.languages[self.resolve(lang)]
        if key in texts:
            return texts[key]
        main = self.languages[DEFAULT_LANG]
        if key not in main:
            raise MissingText(f"{DEFAULT_LANG}.json has no key {key!r}")
        return main[key]

    def text(self, key: str, lang: str | None = DEFAULT_LANG, **params) -> str:
        """The text of key with each {name} replaced by str(params[name])."""
        template = self.raw(key, lang)

        def fill(m: re.Match) -> str:
            name = m.group(1)
            if name not in params:
                raise TextError(f"{key} needs the value {{{name}}}")
            return str(params[name])

        return PLACEHOLDER.sub(fill, template)
