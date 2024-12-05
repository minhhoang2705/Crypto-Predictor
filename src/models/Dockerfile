FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install --no-cache-dir mlflow==2.8.1 psycopg2-binary

# Create directory for artifacts
RUN mkdir -p /mlflow/artifacts

# Set environment variables
ENV MLFLOW_TRACKING_URI=postgresql://mlflow:mlflow@db:5432/mlflow
ENV MLFLOW_ARTIFACTS_DESTINATION=/mlflow/artifacts

# Expose the port
EXPOSE 5000

# Start MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "${MLFLOW_TRACKING_URI}", \
     "--default-artifact-root", "${MLFLOW_ARTIFACTS_DESTINATION}"] 