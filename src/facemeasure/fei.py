"""FEI Face Database: file names, download links and pose numbers (model.md section 5.1).

FEI has 200 people with 14 photos each, 640x480, named <person>-<number>.jpg with the
number written in two digits, for example 1-01.jpg. The zips unpack into
data/raw/fei/images/ (model/scripts/profile_check.py download).

FEI may be used for research only and must not be shared or copied, so FEI photos
and anything drawn over them stay in data/: never in a report, a slide or git.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from facemeasure import paths

PEOPLE = 200
POSES = 14
IMAGE_SIZE = (640, 480)  # width, height
IMAGES_FOLDER = "images"
CHECKSUMS_NAME = "checksums.txt"
ZIP_URL = "https://fei.edu.br/~cet/originalimages_part{part}.zip"
ZIP_PARTS = (1, 2, 3, 4)
NAME = re.compile(r"^([1-9]\d{0,2})-(\d{2})\.jpg$", re.IGNORECASE)

# Pose numbers, found with Face Landmarker on the first five people and confirmed by the
# user on data/outputs/fei_numbering.jpg (profile_check.py). Numbers 1 to 10 turn the head from
# one side to the other; 13 and 14 are frontal photos under other lighting.
PROFILE_FULL = (1, 10)  # full profiles: the face points to the image left, then to the image right
PROFILE_SECOND = (2, 9)  # the next angles, image left then image right
FRONTAL_NEUTRAL = 11
FRONTAL_SMILE = 12


@dataclass(frozen=True)
class FeiName:
    person: int  # 1-200
    number: int  # pose number 1-14


def zip_name(part: int) -> str:
    return f"originalimages_part{part}.zip"


def zip_url(part: int) -> str:
    return ZIP_URL.format(part=part)


def parse_name(name: str) -> FeiName | None:
    """Person and pose number of an FEI file name, or None for any other name."""
    m = NAME.match(name)
    if m is None:
        return None
    person, number = int(m.group(1)), int(m.group(2))
    if not (1 <= person <= PEOPLE and 1 <= number <= POSES):
        return None
    return FeiName(person, number)


def image_name(person: int, number: int) -> str:
    return f"{person}-{number:02d}.jpg"


def images_dir() -> Path:
    return paths.path("fei") / IMAGES_FOLDER


def image_path(person: int, number: int) -> Path:
    return images_dir() / image_name(person, number)
