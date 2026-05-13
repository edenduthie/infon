# Fast DuckDB pipeline for AWS Batch
# Replaces the 60GB/4hr LanceDB IngestAndNormalize job with a <5min DuckDB pipeline.
#
# Build:
#   docker build -t sirius-fast-pipeline -f step_function/batch/fast_pipeline/Dockerfile .
# Run locally:
#   docker run -e NUEXTRACT_S3_BUCKET=sirius-dimefiled-results sirius-fast-pipeline

FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip && \
    pip install \
    duckdb>=1.2.0 \
    orjson>=3.10.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0 \
    networkx>=3.2 \
    model2vec>=0.7.0 \
    openpyxl>=3.1.0 \
    boto3>=1.34.0

RUN useradd -m appuser
WORKDIR /app

# Copy the pipeline code
COPY local_pipeline.py /app/local_pipeline.py
COPY location_gazetteer.py /app/location_gazetteer.py
COPY sbl-pipeline/step_function/batch/fast_pipeline/batch_entrypoint.py /app/batch_entrypoint.py

HEALTHCHECK NONE
USER appuser
ENTRYPOINT ["python", "/app/batch_entrypoint.py"]
