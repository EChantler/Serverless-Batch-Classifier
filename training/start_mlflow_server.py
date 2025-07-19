#!/usr/bin/env python3
"""
MLflow server startup script with SQL backend and S3 artifact storage
"""

import os
import subprocess
import sys
from mlflow_remote_config import MLflowRemoteConfig

def start_mlflow_server():
    """Start MLflow server with remote configuration"""
    
    # Initialize configuration
    config = MLflowRemoteConfig()
    
    # Get database URI
    db_type = os.getenv('DB_TYPE', 'postgresql')
    tracking_uri = config.get_database_uri(db_type)
    
    # Get artifact URI
    bucket_name = os.getenv('MLFLOW_S3_BUCKET', 'mlflow-artifacts-bucket')
    artifact_uri = config.get_artifact_uri(bucket_name)
    
    # Server configuration
    host = os.getenv('MLFLOW_SERVER_HOST', '0.0.0.0')
    port = os.getenv('MLFLOW_SERVER_PORT', '5000')
    
    # Build MLflow server command
    cmd = [
        sys.executable, '-m', 'mlflow', 'server',
        '--backend-store-uri', tracking_uri,
        '--default-artifact-root', artifact_uri,
        '--host', host,
        '--port', port
    ]
    
    print("Starting MLflow server with the following configuration:")
    print(f"  Backend Store URI: {tracking_uri}")
    print(f"  Artifact Root: {artifact_uri}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Access at: http://{host}:{port}")
    print()
    
    # Start the server
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMLflow server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_mlflow_server()
