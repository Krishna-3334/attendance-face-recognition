# Google Cloud Run image for the BioAccess recognition backend.
#   * must listen on $PORT (8080 by default), not a fixed port
#   * filesystem is ephemeral and per-instance, so deploy --max-instances=1
#   * weights are baked in at build time; a 300 MB cold-start download would
#     blow the startup probe
# Measured peak resident memory: ~670 MB. Deployed with 2 GiB for headroom.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --user cython numpy \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts

RUN python scripts/download_models.py

ENV DATA_DIR=$HOME/data \
    DB_PATH=$HOME/data/attendance.db \
    PORT=8080
RUN mkdir -p $HOME/data

EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
