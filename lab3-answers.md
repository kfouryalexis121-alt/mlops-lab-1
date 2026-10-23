# Lab 3 answers

## 1. Registered version and logged artifacts

The best Lab 2 run, `f969a20016df4031b39cf615cf30c828`, was registered as
`food11`, version **1**. Its status is `READY`.

The logged model is an artifact produced by a particular training run. A
registered model adds a stable name and numbered versions, linking those
versions back to the original run. Here, version 1 points to logged model
`m-e113c67e46814967a90386e3e3fd69a9` from that winning run.

## 2. Aliases and model versioning

Aliases such as `champion` and `challenger` are user-defined names pointing to
model versions. They replace workflows based on the deprecated built-in
stages; those particular alias names are conventions, not mandatory stages.

Separately versioning models lets us select releases from training experiments
without changing the experiment history. The `champion` alias currently points
to version 1. We can move that alias to a different version, or move it back
after a regression, while consumers keep using the same model URI.

## 3. Loading through the registry

The API loads `models:/food11@champion` once during startup using
`mlflow.pyfunc.load_model`. MLflow resolves the alias, finds the version's
artifacts, downloads them, and loads the model using its recorded flavor.
There is no machine-specific `.pth` path in the serving code.

To serve a compatible new model, register a new version, assign `champion` to
it, and restart the API/container. Its Python code and Docker image can stay
the same. Changing the alias does not replace the model already loaded in a
running process.

## 4. Docker build cache

The Dockerfile copies `pyproject.toml` and `uv.lock` before copying `src/`.
Dependency installation therefore depends on the dependency files rather than
every source file. Editing only `serve.py` leaves the dependency layers cached;
Docker updates the source layer and subsequent runtime metadata.

The builder runs `uv sync --frozen --no-dev --no-install-project`. The final
option is necessary because the source has not been copied yet, and the API
imports the copied source directly. A BuildKit cache mount keeps downloaded
wheels outside the image layers. See the
[official uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

## 5. Image sizes and layers

The main Dockerfile uses a full Python builder and a slim runtime.
`Dockerfile.single-stage` provides a comparison with the same locked
dependencies while retaining the full Python base and uv.

Both images built successfully on Docker Desktop's Linux/containerd engine.
`docker image ls food11-api` reported:

| Image | Reported local size |
|---|---:|
| `food11-api:latest` (multi-stage) | 2.04 GB |
| `food11-api:single-stage` | 3.38 GB |

The multi-stage image is about **1.34 GB smaller**, approximately **40%** less
by that measurement. These are cumulative local image sizes, so shared layers
mean they should not be added together to estimate unique disk use.

`docker history` showed the dependency environment as the biggest layer in
both images, about **1.45 GB**. The single-stage image additionally retained
the full Python base's development/toolchain layers (including a 619 MB layer)
and the 54.9 MB uv binary. The multi-stage runtime leaves those behind in the
builder.

This Docker installation uses the containerd image store, which retains both
compressed and unpacked layers. Consequently, the image-inspection content
sizes differ from the local-size display: `.Size` reported 435,332,099 bytes
for the multi-stage image and 786,529,949 bytes for the single-stage image.
See [Docker's storage documentation](https://docs.docker.com/engine/storage/containerd).

Linux resolves CPU-only PyTorch wheels from an explicit PyTorch index, avoiding
CUDA dependencies. Torch and torchvision are pinned to the versions used for
training. See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 6. Why .dockerignore matters

Without `.dockerignore`, local environments, datasets, model stores, Git/DVC
metadata, caches, and logs become eligible for the build context. This can
increase transfer time and expose unnecessary files to the builder. If a
Dockerfile uses `COPY . .`, those files can also enlarge the final image.

The most dangerous accidental copy is the host `.venv`: Windows executables,
paths, and native packages cannot replace the Linux virtual environment.
Merely sending an extra folder does not inherently break a build. With this
Dockerfile's explicit dependency-file and `src/` copies, extra context would
primarily waste transfer/storage rather than overwrite the environment.

## 7. Host networking

Inside a normally isolated container, `127.0.0.1` refers to the container's own
loopback interface. It does not refer to the Windows computer running MLflow.
Docker Desktop provides `host.docker.internal` as a hostname resolving to the
host's internal address, so the container uses:

`MLFLOW_TRACKING_URI=http://host.docker.internal:5000`

MLflow listens on `0.0.0.0:5000` and explicitly allows that hostname. On Linux,
the lab's `--network host` option shares host networking instead, allowing the
host's `127.0.0.1:5000` to be used.

## 8. Recreating the container

The application and dependencies are baked into the image. The model is
resolved through `champion` and fetched from MLflow each time a new API process
starts. No model files, database, or dataset are copied into the image or
mounted into the container.

We stopped and removed container `8a3b262b23eb`, then created a new container,
`60368ec756a0`, without running `docker build` again. Both used exactly the
same image, `sha256:0ce3e0486ef541afad9b0a94a77ca94a1f2fb25715896462a624d02c4eee47fb`,
and both had an empty mount list.

The new container completed startup, returned `{"status":"ok"}` from
`/health`, and produced the same prediction and confidence as the previous
container for the sample validation image. This confirms that a fresh
container can resolve the registry alias and fetch the model at runtime from
the host server without rebuilding or mounting the host model directory.

## 9. Sharing the exact image

The Dockerfile is versioned in Git, but a locally built image is only in this
machine's Docker image store. To let another machine pull it, we still need to
tag and push it to a container registry such as GHCR or Docker Hub, configure
access, and deploy using its immutable image digest. A mutable `latest` tag
alone does not identify the exact image reliably.

That machine must also be able to reach the MLflow server and its model
artifacts. `host.docker.internal` identifies the Docker host, so deployment on
another machine needs the appropriate reachable tracking URI. The registry
database and artifacts also need their own persistence/backup arrangements;
they are not included in Git or the application image.

## Local artifact-store migration

The repository directory was renamed from `mlops-lab-1` to `mlops-labs` after
training. Fifteen MLflow experiment/run/logged-model artifact locations still
referenced the missing original folder. Before starting the Lab 3 server, we
backed up `mlflow.db` to an ignored `mlflow.db.before-lab3-*.bak` file and changed
those locations to the matching `mlflow-artifacts:/...` paths.

The server now uses `--serve-artifacts --artifacts-destination ./mlruns`, serving
the existing model files over HTTP. We preserved the original runs, metrics,
model files, and run IDs. This also avoids requiring a Linux container to read
Windows filesystem paths. Changing server flags alone does not migrate
existing experiments; see the
[MLflow tracking-server documentation](https://www.mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

## Verification

- Registration was repeated successfully and reused version 1.
- The serving preprocessing matched the training transform exactly for a
  processed validation image.
- All six API integration tests passed locally, covering health, real-image
  predictions, repeated predictions, grayscale/RGBA PNGs, invalid and empty
  files, missing uploads, and oversized uploads.
- The same six checks passed against the Docker container.
- A fresh container from the same image passed health and prediction checks.
- Local and Docker predictions for the same processed image agreed within
  floating-point rounding (confidence difference less than 0.000001).
- Both image builds and the frozen lockfile check completed successfully.
