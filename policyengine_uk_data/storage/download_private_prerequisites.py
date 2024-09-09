from policyengine_uk_data.utils.github import download
from pathlib import Path

FOLDER = Path(__file__).parent

download(
    "PolicyEngine",
    "ukda",
    "release",
    "frs_2020_21.zip",
    FOLDER / "frs_2020_21.zip",
)
download(
    "PolicyEngine",
    "ukda",
    "release",
    "frs_2021_22.zip",
    FOLDER / "frs_2021_22.zip",
)
download(
    "PolicyEngine",
    "ukda",
    "release",
    "frs_2022_23.zip",
    FOLDER / "frs_2022_23.zip",
)
