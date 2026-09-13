# Hugging Face Spaces (Docker SDK) image for the BioAccess recognition backend.
#
# Two things matter here and are easy to get wrong on Spaces:
#   1. The container runs as UID 1000 with $HOME=/home/user, so every path the
#      app writes to must be owned by that user. Running as root works locally
#      and fails on Spaces.
#   2. Model weights are baked in at build time rather than downloaded on first
#      request, so a visitor never waits ~300 MB for their first probe.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt ./
# cython and numpy first: insightface builds a C extension at install time and
# fails without them present.
RUN pip install --no-cache-dir --user cython numpy \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts
COPY --chown=user eval ./eval

RUN python scripts/download_models.py

ENV DATA_DIR=$HOME/data \
    DB_PATH=$HOME/data/attendance.db
RUN mkdir -p $HOME/data

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
