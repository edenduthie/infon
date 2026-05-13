# Batch container for strategic analysis (counterfactual, implicit, Faustian).
#
# Reads clusters + raw_events from LanceDB, aggregates counterfactual/implicit/
# trade-off signals, calls Bedrock LLM to triangulate strategic patterns, writes
# results to LanceDB strategic_analysis table and S3 JSON.
#
# Build (from step_function/):
#   docker build -f batch/strategic_analysis/Dockerfile -t sirius-strategic-analysis .
#
# Push:
#   aws ecr get-login-password | docker login --username AWS --password-stdin 038462780313.dkr.ecr.us-east-1.amazonaws.com
#   docker tag sirius-strategic-analysis:latest 038462780313.dkr.ecr.us-east-1.amazonaws.com/sirius-strategic-analysis:latest
#   docker push 038462780313.dkr.ecr.us-east-1.amazonaws.com/sirius-strategic-analysis:latest

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    boto3>=1.34.0 \
    lancedb>=0.6.0 \
    pylance \
    pyarrow>=14.0.0 \
    pandas>=2.0.0

RUN useradd -m appuser
WORKDIR /app
COPY batch/strategic_analysis/handler.py /app/handler.py
COPY batch/ingest_normalize/shared/ /app/shared/

HEALTHCHECK NONE
USER appuser
ENTRYPOINT ["python", "/app/handler.py"]
