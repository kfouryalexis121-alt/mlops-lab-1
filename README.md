# Food-11 MLOps labs

One repository for the Food-11 labs: data preparation and DVC (Lab 1),
training and MLflow tracking (Lab 2), model serving with Docker (Lab 3),
Docker Compose (Lab 4), and CI/CD with GitHub Actions (Lab 5).

## Lab documents

| Lab | Instructions | Answers |
|---|---|---|
| 1: Git, DVC, and data preparation | [Instructions](lab/lab1-instructions.md) | [Answers](lab/lab1.md) |
| 2: Training and MLflow | [Instructions](lab/lab2-instructions.md) | [Answers](lab/lab2.md) |
| 3: Docker and model serving | [Instructions](lab/lab3-instructions.md) | [Answers](lab/lab3.md) |
| 4: Docker Compose | [Instructions](lab/lab4-instructions.md) | [Answers](lab/lab4.md) |
| 5: CI/CD with GitHub Actions | [Instructions](lab/lab5-instructions.md) | [Answers](lab/lab5.md) |

## Lab 5: tests, data validation, and CI/CD

Install development dependencies and run the server-independent checks:

```powershell
uv sync --frozen
uv run ruff check src tests
uv run pytest -q
```

Live API tests are kept under `tests/integration` and run with the Compose test
script below. `SKIP_MODEL=1` enables CI startup without MLflow: `/health` returns
200 and `/predict` returns 503 until a model is supplied. Normal deployments
omit this variable and load the registered champion.

The original mini dataset contains cross-split duplicates. Create the separate
clean dataset once, then validate it:

```powershell
uv run python -m src.food11.prepare_lab5_data
uv run python -m src.food11.validate_data data_local/food11_lab5
uv run python -m src.food11.train --dataset lab5 --lr 0.0001 --epochs 5
```

The preparation command refuses to overwrite an existing destination. The
original DVC dataset stays intact; the clean copy is reproducible and ignored
by Git. Training validates the contract before loading data or creating a run.

Register a candidate without moving `champion`, then apply the gate:

```powershell
uv run python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); print(mlflow.register_model('runs:/RUN_ID/model', 'food11').version)"
uv run python -m src.food11.promote --version VERSION --test-dir data_local/food11_lab5/evaluation
```

Both runs must have `test_accuracy` on the same evaluation dataset. The
promotion script records its decision as model-version tags. Restart inference
after a successful promotion to serve the selected model.

GitHub Actions runs `test` then `smoke` on pushes and pull requests. On pushes
to `main`, `push` publishes the tested image as
`ghcr.io/kfouryalexis121-alt/mlops-lab-1:<commit-sha>`. Publishing is separate
from deployment; pull that tag (or its digest) and provide a reachable MLflow
tracking URI when running it. See [Lab 5 answers](lab/lab5.md) for results.

## Lab 4: run the Compose stack

Docker Desktop must be running with Linux containers. Stop the Lab 3 host
MLflow server before using port 5000, or set `$env:MLFLOW_PORT = "5001"` in
PowerShell and use that port in the import command and browser.

On the first run, start MLflow and import the saved Lab 2 model into its new
named volume:

```powershell
docker compose up -d --build --wait mlflow
.\.venv\Scripts\python.exe -m src.food11.import_model --model-dir mlruns/2/models/m-e113c67e46814967a90386e3e3fd69a9/artifacts --source-run-id f969a20016df4031b39cf615cf30c828 --tracking-uri http://127.0.0.1:5000
docker compose up -d --build --wait
```

The import copies model artifacts into the Compose tracking server and records
the original run ID as a tag. It leaves the original Lab 2 tracking store intact
and skips importing if `food11` already has versions. A fresh clone needs access
to the saved model directory; model binaries are not stored in Git.

Open the [upload page](http://127.0.0.1:8501) and
[MLflow UI](http://127.0.0.1:5000). Inference is accessible only inside the Compose
network. Health checks ensure services are ready before dependents start.
The project uses `champion` in place of the lab handout's older `Staging` stage.

After assigning `champion` to a new compatible version, run:

```powershell
docker compose restart inference
docker compose ps
```

`docker compose down` preserves the MLflow volume; `docker compose up -d --wait`
restores the stack. **`docker compose down -v` deletes the Compose database and
models**, so the initial import must be repeated afterward. The host's original
`mlflow.db` and `mlruns/` are separate from this volume.

With the stack running and the mini dataset available, run the API and frontend
checks from PowerShell:

```powershell
.\scripts\test_compose.ps1
```

This copies test fixtures into the running containers, checks the API, and tests
the Streamlit upload flow against inference without publishing the API port.

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
the proxy automatically; see the migration note in [Lab 3 answers](lab/lab3.md).

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
and source code, and runs as user 10001. See [Lab 3 answers](lab/lab3.md) for measured sizes
and answers to the nine lab questions.
