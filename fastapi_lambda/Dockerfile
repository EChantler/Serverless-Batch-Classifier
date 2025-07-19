# Dockerfile for Lambda FastAPI application
FROM public.ecr.aws/lambda/python:3.9

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["app.handler"]
