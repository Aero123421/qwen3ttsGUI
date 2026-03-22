FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/data/hf-cache \
    HF_HUB_CACHE=/data/hf-cache/hub \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    OUTPUT_DIR=/data/outputs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    build-essential \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

RUN python -m pip install flash-attn --no-build-isolation || \
    echo "flash-attn installation failed; the app will fall back to default attention."

COPY app /app/app
COPY README.md /app/README.md

EXPOSE 7860

CMD ["python", "-m", "app.main"]
