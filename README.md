# Food-11 MLOps labs

One repository for the Food-11 labs: data preparation and DVC (Lab 1),
training and MLflow tracking (Lab 2), and model serving with Docker (Lab 3).

## Environment

Install the locked dependencies with `uv sync --frozen`. The project uses
Python 3.10; the Linux Docker image uses CPU-only PyTorch wheels. On Windows,
activate the environment from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
```

The dataset and MLflow database/artifacts are local resources, excluded from
Git and from the Docker build. Cloning this repository alone does not restore
the trained model or registry. Use the existing Lab 2 tracking store, or
restore it before starting the API.

## Lab 3: start MLflow

Run this from the project root in a separate terminal and leave it running:

```powershell
python -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --serve-artifacts --artifacts-destination ./mlruns --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*" --workers 1
```

Open <http://127.0.0.1:5000>. The server proxies artifact downloads from `mlruns/`,
allowing containers to fetch the model without mounting Windows directories.
Existing experiments created with `file:` artifact locations do not switch to
the proxy automatically; see the migration note in `lab3-answers.md`.

Register the best Lab 2 model and assign its `champion` alias:

```powershell
python -m src.food11.register_model --run-id f969a20016df4031b39cf615cf30c828
```

Repeating this command reuses a model version from that run. In this project,
`models:/food11@champion` resolves to version 1.

## Run the API locally

In another terminal with the environment activated:

```powershell
python -m uvicorn src.food11.serve:app --host 127.0.0.1 --port 8000
```

The API loads `models:/food11@champion` once at startup. It reads
`MLFLOW_TRACKING_URI`, defaulting to `http://127.0.0.1:5000`. After changing the
champion alias, restart the API to load the new version.

- `GET /health`: returns `{"status":"ok"}` after successful startup.
- `POST /predict`: accepts an image in the multipart field `file` and returns
  `category` plus a softmax `confidence` score between 0 and 1.
- Interactive API documentation: <http://127.0.0.1:8000/docs>.

Images are converted to RGB, resized to 128 by 128, and normalized as in
training. Class names follow ImageFolder's alphabetical directory order.
Uploads above 10 MiB return HTTP 413, unreadable images return HTTP 400, and
requests without a file return HTTP 422.

Test from PowerShell with the real curl executable:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST -F "file=@data/food11_processed_mini/validation/Bread/0_0.jpg" http://127.0.0.1:8000/predict
python -m unittest discover -s tests -v
```

The checks require the API, MLflow, and the mini dataset. They test real model
predictions, repeated requests, different image modes, and invalid uploads.

## Build and run Docker

Start Docker Desktop using Linux containers. Stop the local API with Ctrl+C
before using the same host port for Docker.

```powershell
docker build -t food11-api:latest .
docker run -d --name food11-api -p 127.0.0.1:8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
docker logs food11-api
```

Wait for `Application startup complete`, then use the same curl commands and
integration checks above. The container downloads the model from MLflow at
startup; neither the model nor the dataset is baked into the image.

On Linux, use host networking instead:

```bash
docker run -d --name food11-api --network host -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 food11-api:latest
```

Recreate the Windows container from the existing image without rebuilding:

```powershell
docker stop food11-api
docker rm food11-api
docker run -d --name food11-api -p 127.0.0.1:8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

The model must remain available from the tracking server. The running service
keeps its loaded model in memory until the process restarts.

## Compare image sizes

The comparison Dockerfile retains the full Python base and uv in one stage:

```powershell
docker build -f Dockerfile.single-stage -t food11-api:single-stage .
docker image ls food11-api
docker history food11-api:latest
docker history food11-api:single-stage
```

The multi-stage runtime contains only the slim base, dependency environment,
and source code, and runs as user 10001. See `lab3-answers.md` for measured sizes
and answers to the nine lab questions.
