from policyengine_uk_data.utils.github import download
from pathlib import Path
import zipfile


def extract_zipped_folder(folder):
    folder = Path(folder)
    with zipfile.ZipFile(folder, "r") as zip_ref:
        zip_ref.extractall(folder.parent)


FOLDER = Path(__file__).parent

FILES = [
    "frs_2022_23.zip",
    "lcfs_2021_22.zip",
    "was_2006_20.zip",
    "etb_1977_21.zip",
    "spi_2020_21.zip",
]

for file in FILES:
    if (FOLDER / file).exists():
        continue
    download(
        "PolicyEngine",
        "ukda",
        "release",
        file,
        FOLDER / file,
    )
    extract_zipped_folder(FOLDER / file)
    (FOLDER / file).unlink()
