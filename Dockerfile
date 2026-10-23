# syntax=docker/dockerfile:1
FROM python:3.10-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
# Source is copied later, so install dependencies without building this project.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM python:3.10-slim-bookworm AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    HOME=/tmp \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
