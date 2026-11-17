# Lab 3 answers

## 1. Registered version and logged artifacts

The best Lab 2 run, `f969a20016df4031b39cf615cf30c828`, was registered as
`food11`, version **1**, with status `READY`. A logged model belongs to a
training run; registration adds a stable name and numbered versions linked
back to that run.

## 2. Aliases and model versioning

Aliases such as `champion` and `challenger` are custom names for model
versions, replacing workflows based on deprecated stages. `champion` points
to version 1. Moving the alias allows upgrades or rollbacks while keeping
the same model URI and preserving training history.

## 3. Loading through the registry

At startup, the API loads `models:/food11@champion` using
`mlflow.pyfunc.load_model`. MLflow resolves the alias and fetches the model.
To deploy a compatible update, register it, move `champion`, and restart the
API/container. No code or image change is needed, but changing the alias
alone does not update an already running process.

## 4. Docker build cache

The Dockerfile copies `pyproject.toml` and `uv.lock` before `src/`, so editing
`serve.py` reuses the cached dependency layers. The builder uses
`uv sync --frozen --no-dev --no-install-project` because source files are not
yet copied. A BuildKit cache mount also reuses downloaded wheels.

## 5. Image sizes and layers

Both images built successfully. `docker image ls food11-api` reported:

| Image | Reported local size |
|---|---:|
| `food11-api:latest` (multi-stage) | 2.04 GB |
| `food11-api:single-stage` | 3.38 GB |

The multi-stage image is **1.34 GB smaller**, about **40%**, because its slim
runtime leaves behind builder tools and the full Python base. Dependencies
were the largest layer in both images (about 1.45 GB). CPU-only PyTorch wheels
avoid CUDA dependencies and match the training versions.

These local sizes include containerd storage overhead and shared layers;
they differ from image content sizes and should not be added to estimate
unique disk usage.

## 6. Why .dockerignore matters

`.dockerignore` excludes local environments, datasets, model stores, metadata,
and caches from the build context, reducing transfers and accidental copies.
Copying a Windows `.venv` could break the Linux environment. Here, explicit
file copies mainly limit extra context to wasted transfer and storage.

## 7. Host networking

Inside an isolated container, `127.0.0.1` refers to the container itself.
On Docker Desktop, the API reaches the host MLflow server through:

`MLFLOW_TRACKING_URI=http://host.docker.internal:5000`

MLflow listens on `0.0.0.0:5000` and allows that hostname. On Linux,
`--network host` shares the host network, allowing `127.0.0.1:5000` instead.

## 8. Recreating the container

The image contains the application and dependencies; each API startup fetches
the model from MLflow through `champion`. We removed and recreated the
container using the same image, without rebuilding or mounting model files.
The new container returned `{"status":"ok"}` from `/health` and reproduced
the previous prediction and confidence.

## 9. Sharing the exact image

To share the locally built image, tag and push it to GHCR or Docker Hub,
configure access, and deploy by immutable digest. A `latest` tag can change.
The other machine also needs a reachable MLflow server and artifacts;
`host.docker.internal` always refers to its own Docker host. MLflow data
requires separate persistence and backups.

## Local artifact-store migration

After the repository folder was renamed, 15 MLflow artifact locations still
pointed to the old path. We backed up `mlflow.db` and updated those locations
to `mlflow-artifacts:/...`, preserving runs, metrics, models, and run IDs.
The server uses `--serve-artifacts --artifacts-destination ./mlruns` so Linux
containers can fetch artifacts over HTTP instead of using Windows paths.

## Verification

- Repeated registration reused version 1, and preprocessing matched training.
- All six API integration tests passed locally and in Docker.
- A recreated container passed health and prediction checks.
- Local and Docker confidence values differed by less than 0.000001.
- Both image builds and the frozen lockfile check passed.
