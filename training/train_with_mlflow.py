#!/usr/bin/env python3
"""
MLflow-enabled training script for defect classification
"""

import os
import mlflow
import mlflow.tensorflow
import pymysql
from train_defect_classifier import DefectClassifier
from mlflow_remote_config import MLflowRemoteConfig
DB_TYPE = "mysql"  # or 'sqlite'
DB_NAME = "mlflow"
DB_HOST = "batch-classifier-db.ckx6ieea6u2k.us-east-1.rds.amazonaws.com"
DB_PORT = 3306
DB_USER = "admin"
DB_PASSWORD = "MySecurePassword123!"
AWS_REGION = "us-east-1"  # Change to your AWS region
S3_BUCKET_NAME = "serverless-batch-classifier-bucket"

def ensure_database_exists(host, port, user, password, db_name):
    try:
        conn = pymysql.connect(host=host, port=port, user=user, password=password)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
        conn.close()
    except Exception as e:
        print(f"Database creation error: {e}")

def main():
    """Main training function with MLflow tracking"""
    
    # Configure MLflow for remote backend (SQL + S3)
    # Comment out the next 4 lines to use local MLflow
    remote_config = MLflowRemoteConfig()
    tracking_uri, artifact_uri = remote_config.configure_mlflow(
        db_type=DB_TYPE,  # or 'mysql', 'sqlite'
        bucket_name=S3_BUCKET_NAME,
        region_name=AWS_REGION,  # Change to your AWS region
        db_host=DB_HOST,  # Replace with your database host
        db_port=DB_PORT,  # Default MySQL port
        db_user=DB_USER,  # Replace with your database user
        db_password=DB_PASSWORD  # Replace with your database password
    )
    ensure_database_exists(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db_name=DB_NAME
    )
    # Alternative: Use local MLflow (comment out remote_config above)
    # mlflow.set_tracking_uri("http://localhost:5000")  # For local MLflow server
    
    # Configuration
    config = {
        "image_size": (224, 224),
        "batch_size": 32,
        "epochs": 3,
        "experiment_name": "defect-classification-experiments"
    }
    
    # Initialize classifier with remote artifact location
    classifier = DefectClassifier(
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        experiment_name=config["experiment_name"],
        artifact_uri=artifact_uri
    )
    
    # Create model
    model = classifier.create_model()
    print("Model architecture:")
    model.summary()
    
    # Check if data exists
    data_dir = "data/train"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        print("Please ensure your data is in the correct structure:")
        print("  data/")
        print("    train/")
        print("      defective/")
        print("      normal/")
        print("    test/")
        print("      defective/")
        print("      normal/")
        return
    
    # Train with MLflow tracking
    print("\nStarting training with MLflow tracking...")
    run_name = f"defect-classifier-{config['epochs']}-epochs"
    
    try:
        # Train the model
        history = classifier.train(
            data_dir=data_dir,
            epochs=config["epochs"],
            run_name=run_name
        )
        
        # Verify model registration
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        try:
            registered_models = client.search_registered_models()
            print(f"\nRegistered models after training:")
            for model in registered_models:
                print(f"  - {model.name}")
                latest_versions = client.get_latest_versions(model.name, stages=["Production", "Staging", "None"])
                for version in latest_versions:
                    print(f"    Version {version.version}, Stage: {version.current_stage}")
                    print(f"    Source: {version.source}")
        except Exception as e:
            print(f"Error checking registered models: {e}")
        
        # Evaluate on test data if available
        test_dir = "data/test"
        if os.path.exists(test_dir):
            print("Evaluating on test data...")
            
            # Start a new run for evaluation or continue the existing one
            if not mlflow.active_run():
                mlflow.start_run(run_name=f"{run_name}-evaluation")
            
            report, cm, predictions = classifier.evaluate(test_dir, log_to_mlflow=True)
            print("Classification Report:")
            print(report)
            
            # Close MLflow run if we started one
            if mlflow.active_run():
                mlflow.end_run()
        
        # Save the model (native Keras format)
        classifier.save_model("defect_classifier_model.keras")
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved as 'defect_classifier_model.keras'")
        print(f"Check MLflow UI to view experiment results:")
        print(f"  mlflow ui")
        print(f"  Then navigate to http://localhost:5000")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise

def run_mlflow_ui(tracking_uri, artifact_uri, host="127.0.0.1", port=5000):
    """Helper function to start MLflow server UI with remote backend"""
    import subprocess
    import sys
    try:
        print("Starting MLflow server...")
        print(f"Backend store URI: {tracking_uri}")
        print(f"Default artifact root: {artifact_uri}")
        print(f"Navigate to http://{host}:{port} to view experiments")
        subprocess.run([
            sys.executable, "-m", "mlflow", "server",
            "--backend-store-uri", tracking_uri,
            "--default-artifact-root", artifact_uri,
            "--host", host,
            "--port", str(port)
        ], check=True)
    except KeyboardInterrupt:
        print("\nMLflow server stopped.")
    except Exception as e:
        print(f"Error starting MLflow server: {str(e)}")

if __name__ == "__main__":
    import sys
    # Run MLflow server UI with remote backend
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        # Ensure database and S3 are configured
        ensure_database_exists(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            db_name=DB_NAME
        )
        remote_config = MLflowRemoteConfig()
        tracking_uri, artifact_uri = remote_config.configure_mlflow(
            db_type=DB_TYPE,
            bucket_name=S3_BUCKET_NAME,
            region_name=AWS_REGION,
            db_host=DB_HOST,
            db_port=DB_PORT,
            db_user=DB_USER,
            db_password=DB_PASSWORD
        )
        run_mlflow_ui(tracking_uri, artifact_uri)
    else:
        main()
