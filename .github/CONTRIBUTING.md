## Updating data

If your changes present a non-bugfix change to one or more datasets which are cloud-hosted (FRS and EFRS), then please change both the filename and URL (in both the class definition file and in `storage/upload_completed_datasets.py`). This enables us to store historical versions of datasets separately and reproducibly.

## Updating the versioning

Please add to `changelog.yaml` and then run `make changelog` before committing the results ONCE in this PR.
