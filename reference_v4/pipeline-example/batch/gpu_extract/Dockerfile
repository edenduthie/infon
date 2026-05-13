# GPU extraction job for AWS Batch
# Runs NuExtract-2.0-4B via vLLM on g5.xlarge (A10G 24GB GPU)
#
# Build:
#   docker build -t sirius-gpu-extract .
# Run locally:
#   docker run --gpus all -e SQS_QUEUE_URL=... sirius-gpu-extract

FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Create venv with Python 3.11
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch + vLLM + dependencies
RUN pip install --upgrade pip && \
    pip install vllm dateparser beautifulsoup4 lxml boto3

# Copy extraction scripts
RUN useradd -m appuser
WORKDIR /workspace
COPY gpu_extract.py nuextract_template.py warc_fetch.py postprocess.py ./
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Model weights downloaded at startup (not baked in — keeps image ~10GB vs 27GB)
HEALTHCHECK NONE
USER appuser
ENTRYPOINT ["./entrypoint.sh"]
