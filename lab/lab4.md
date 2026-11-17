# Lab 4 answers

This implementation uses the `champion` alias instead of the handout's older
`Staging` stage, matching Lab 3.

## 1. Running without a volume

Files written inside a container survive stopping it, but disappear when that
container is removed. A new MLflow container starts with an empty database:
previous experiments, runs, and registered models are absent.

## 2. Named volume versus bind mount

A named volume is managed by Docker and does not depend on a particular host
folder. A bind mount also persists data, but requires a suitable host path and
permissions. Neither alone protects against losing the host machine.

## 3. Resolving the mlflow hostname

Compose connects the services to a shared network. Docker's internal DNS maps
the service name `mlflow` to its container, so inference can use
`http://mlflow:5000`.

## 4. Configuring the inference URL

An environment variable lets the same frontend image use different API
addresses in Compose, standalone Docker, or local development. The hostname
`inference` only resolves on the appropriate Docker network.

## 5. Keeping inference internal

Only the browser-facing MLflow UI and frontend need published ports. The
frontend's Python process calls `http://inference:8000` over the Compose
network; the browser does not contact inference directly.

## 6. Startup readiness

With plain `depends_on`, inference can start before MLflow is ready. Loading
the model then fails and the API process exits. This Compose file uses health
checks and `condition: service_healthy` to wait for readiness. A healthy but
empty MLflow server still needs a model imported before inference can start.
See [Docker's startup-order documentation](https://docs.docker.com/compose/how-tos/startup-order/).

## 7. Published ports

The configuration publishes MLflow on `127.0.0.1:5000` and the frontend on
`127.0.0.1:8501`. Inference exposes container port 8000 without a host mapping.
The host ports can be overridden with `MLFLOW_PORT` and `FRONTEND_PORT`.

## 8. Loading a new model version

Refreshing the frontend still uses the model already loaded by inference.
After moving `champion` to the new version, run
`docker compose restart inference`. The API resolves the alias at startup
and logs the exact version it loaded.

In the check, version 2 reused version 1's artifacts. Logs confirmed version 1
before restart and version 2 afterward; identical weights need not change the
prediction.

## 9. Why no rebuild is needed

The image contains serving code and dependencies. Model artifacts are fetched
from MLflow at startup, so a compatible model update only needs a restart.
A model requiring different dependencies may also require an image rebuild.

## 10. Persistence

`docker compose down` removes containers but preserves the named volume,
including the registry, aliases, and artifacts. `docker compose down -v`
also deletes that volume, so the next startup has an empty registry and needs
another model import. The original host Lab 2 store remains separate.

## 11. Scaling and machine failure

Compose can run three inference replicas on one host, but needs a proxy/load
balancer to distribute requests. It does not provide multi-host scheduling
or recovery after host failure. That requires an orchestrator such as
Kubernetes, plus durable shared artifact storage, an external database,
backups, and a recovery or high-availability strategy for MLflow.

## Implementation notes

MLflow serves artifacts over HTTP from its named volume, so inference does
not need a shared filesystem mount. The import command uploads the saved
Lab 2 model and records its original run ID, while preserving the host store.
See the [MLflow tracking-server documentation](https://mlflow.org/docs/latest/tracking/server/).

## Verified results

- All three images built and all three services became healthy.
- Six API integration checks, two startup checks, and four frontend checks passed.
- The frontend test uploaded a real validation image through the Compose network
  and displayed `Meat (59.1%)` for `validation/Bread/0_0.jpg` (a misclassification).
- Restarting inference loaded version 2 without rebuilding an image.
- Ordinary `down/up` preserved version 2, its alias, and its downloadable artifacts.
- `down -v` cleared the registry. Reimporting restored version 1 and `champion`;
  all 12 checks then passed again.
- An experiment created in a standalone container without a volume disappeared
  after removing and replacing that container.
- The final stack is healthy, with MLflow on port 5000, the frontend on 8501,
  and inference available only inside the Docker network.
