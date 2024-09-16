from policyengine_uk_data.utils.github import download
from policyengine_uk_data.data_storage import STORAGE_FOLDER

PREREQUISITES = [
    {
        "repo": "ukda",
        "file_name": "frs_2022_23.h5",
    },
    {
        "repo": "ukda",
        "file_name": "enhanced_frs_2022_23.h5",
    },
]


def download_data():
    for prerequisite in PREREQUISITES:
        if not (STORAGE_FOLDER / prerequisite["file_name"]).exists():
            download(
                "PolicyEngine",
                prerequisite["repo"],
                "release",
                prerequisite["file_name"],
                STORAGE_FOLDER / prerequisite["file_name"],
            )
