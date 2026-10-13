# syntax=docker/dockerfile:1
# NOTE: this Dockerfile was written carefully but has NOT been built locally
# (Docker is not available on the development machine).

FROM python:3.11-slim AS build
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock pyproject.toml README.md ./
COPY src ./src
# Runtime dependencies only (the dev tools in the lock file are skipped), with
# every version constrained by the committed lock file.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --constraint requirements.lock .

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IDE_DATA_DIR=/data \
    IDE_LOG_LEVEL=INFO
RUN groupadd --system app && useradd --system --gid app --home /app app \
    && mkdir -p /data && chown app:app /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "idengine.api.app:build_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
