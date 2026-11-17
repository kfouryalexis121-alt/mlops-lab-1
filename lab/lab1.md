# Lab 1 answers

## 1. Files created by uv

`pyproject.toml` defines the project, dependencies, and build settings.
`.python-version` selects the Python version, `README.md` describes the project,
and the Python source provides its starting code. In this packaged project,
the entry point is in `src/mlops_lab_1/__init__.py`. `uv.lock` records exact
dependency versions when dependencies are resolved.

## 2. Files created by DVC

`dvc init` creates `.dvc/` and `.dvcignore`. `.dvc/config` holds project settings,
`.dvc/.gitignore` excludes local DVC files, and `.dvcignore` lists files DVC
should skip. Commit these configuration files to Git, but exclude the data
cache, temporary files, and private `.dvc/config.local` settings.

## 3. Credential storage

With `--global`, credentials go into the user's DVC configuration, normally
`%LocalAppData%\iterative\dvc\config` on Windows. Other scopes are `--local`
(`.dvc/config.local`), project scope (the default, `.dvc/config`), and
`--system` (machine-wide). Credentials must not be pushed to GitHub; use
local or user-level configuration. See the [DVC configuration documentation](https://doc.dvc.org/user-guide/project-structure/configuration).

## 4. Changes to .gitignore

`dvc add data` added `/data` to `.gitignore`, keeping the actual dataset out
of Git. Git tracks the small DVC pointer instead, while DVC manages the data.

## 5. Contents of data.dvc

`data.dvc` records the dataset path, content hash, total size, and file count.
The current pointer describes 36,578 files totaling 1,320,320,598 bytes.
Its hash identifies the dataset version; the remote address is stored
separately in `.dvc/config`.

## 6. GitHub and DagsHub contents

GitHub contains the code, `data.dvc`, and DVC configuration, but not the images.
The configured DVC remote is `https://dagshub.com/kfouryalexis121/mlops-lab-1.dvc`.
`dvc push` uploads the actual data there. The configuration confirms the remote,
but does not prove that the upload succeeded; the DagsHub UI was not verified
while preparing these answers.

## 7. Restoring data after cloning

A fresh Git clone contains the code and pointer files, but no dataset.
Run `dvc pull` with access to the configured remote to download the referenced
data and populate `data/`. See the [DVC pull documentation](https://doc.dvc.org/command-reference/pull).

## 8. Switching dataset versions

Commit `52e7ccd` tracks the raw dataset; `059206d` adds the processed datasets.
Checking out `52e7ccd` and running `dvc checkout` should remove
`food11_processed` and `food11_processed_mini`, restoring the raw-only version.
Returning to `main` and running `dvc checkout` restores the processed folders.
This requires the corresponding data in the local cache; use `dvc pull` if
it is missing. This describes the expected result from the committed pointers,
not a checkout experiment rerun for this answer file.

## Data-storage approach

The project is configured to use DagsHub, with no local-remote fallback in the
committed configuration. The preprocessing script creates 128x128 images in
class folders and a mini dataset with at most 100 images per class per split.
The current DVC pointer tracks the combined dataset, rather than only the mini
subset.
