# Batch container for sequential pattern mining + type construction.
#
# Runs mine_patterns.py against the SQLite hypergraph produced by BuildSQLite.
# Mines sequential patterns (PrefixSpan), builds causal sequences, detects
# temporal motifs, runs VF2 matching, enriches the hypergraph with new node
# types (SkeletalPlay, CausalSequence, Play), and exports V2 tables as Parquet.
#
# Build (from step_function/):
#   docker build -f batch/mine_patterns/Dockerfile -t sirius-mine-patterns .
#
# Push:
#   aws ecr get-login-password | docker login --username AWS --password-stdin 038462780313.dkr.ecr.us-east-1.amazonaws.com
#   docker tag sirius-mine-patterns:latest 038462780313.dkr.ecr.us-east-1.amazonaws.com/sirius-mine-patterns:latest
#   docker push 038462780313.dkr.ecr.us-east-1.amazonaws.com/sirius-mine-patterns:latest

FROM python:3.12-slim

RUN pip install --no-cache-dir \
    boto3>=1.34.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    pyarrow>=14.0.0 \
    prefixspan>=0.5.2 \
    networkx>=3.0

RUN useradd -m appuser
WORKDIR /app
COPY lambdas/mine_patterns.py /app/mine_patterns.py

HEALTHCHECK NONE
USER appuser
ENTRYPOINT ["python", "/app/mine_patterns.py"]
