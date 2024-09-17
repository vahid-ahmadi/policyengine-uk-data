from policyengine_uk_data.utils.github import upload
from pathlib import Path
from tqdm import tqdm

FOLDER = Path(__file__).parent

FILES = [
    "frs_2022_23.h5",
    "enhanced_frs_2022_23.h5",
    "extended_frs_2022_23.h5",
    "reweighted_frs_2022_23.h5",
]

for file in tqdm(FILES):
    upload(
        "PolicyEngine",
        "ukda",
        "release",
        file,
        FOLDER / file,
    )
