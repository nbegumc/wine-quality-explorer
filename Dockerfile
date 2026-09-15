# syntax=docker/dockerfile:1
# Reference environment for the recorded results: the locked Python dependencies on one
# pinned Linux base, run as linux/amd64 (see compose.yaml for why the architecture matters).
#   docker compose up                       -> dashboard at http://localhost:8000
#   docker compose run --rm experiment      -> regenerate every artifact into the working tree
FROM ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim

WORKDIR /project

# pyAgrum's Linux wheel links against the OpenMP runtime, which the slim image omits.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HOST=0.0.0.0 \
    PORT=8000

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync

EXPOSE 8000
CMD ["uv", "run", "serve.py"]
