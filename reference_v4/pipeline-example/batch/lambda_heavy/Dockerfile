# Heavy Lambda container for functions needing lancedb + model2vec.
#
# Used by: build_sqlite, export_workbook, mine_patterns
#
# Each function is deployed as a separate Lambda pointing to this image,
# with CMD overridden to select the handler module.
#
# Build (from step_function/):
#   docker build -f batch/lambda_heavy/Dockerfile -t sirius-pipeline-lambda-heavy .

FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY batch/lambda_heavy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model2vec model during build so Lambda doesn't hit read-only fs
ENV HF_HOME=/tmp/hf_cache
RUN python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"
RUN mv /tmp/hf_cache ${LAMBDA_TASK_ROOT}/hf_cache
ENV HF_HOME=${LAMBDA_TASK_ROOT}/hf_cache

# Copy handler modules
COPY lambdas/build_sqlite.py ${LAMBDA_TASK_ROOT}/build_sqlite.py
COPY lambdas/export_workbook.py ${LAMBDA_TASK_ROOT}/export_workbook.py
COPY lambdas/mine_patterns.py ${LAMBDA_TASK_ROOT}/mine_patterns.py

# Run as non-root
RUN dnf install -y shadow-utils && useradd -m appuser && dnf clean all
HEALTHCHECK NONE
USER appuser

# Default handler (override per-function via Lambda CMD config)
CMD ["build_sqlite.handler"]
