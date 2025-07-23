# Dockerfile for Lambda FastAPI application
FROM public.ecr.aws/lambda/python:3.9

# Set TensorFlow environment variables for CPU-only execution and reduced logging
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONUNBUFFERED=1

# Set MLflow environment variables to prevent local directory creation
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/tmp/mlflow-artifacts
ENV MLFLOW_ARTIFACT_ROOT=/tmp/mlflow-artifacts
ENV TMPDIR=/tmp
ENV TMP=/tmp
ENV TEMP=/tmp

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["app.handler"]
