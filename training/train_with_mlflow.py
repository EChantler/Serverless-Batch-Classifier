#!/usr/bin/env python3
"""
MLflow-enabled training script for defect classification
"""

import os
import mlflow
import mlflow.tensorflow
from train_defect_classifier import DefectClassifier
from mlflow_remote_config import MLflowRemoteConfig

def main():
    """Main training function with MLflow tracking"""
    
    # Configure MLflow for remote backend (SQL + S3)
    # Comment out the next 4 lines to use local MLflow
    remote_config = MLflowRemoteConfig()
    tracking_uri, artifact_uri = remote_config.configure_mlflow(
        db_type='postgresql',  # or 'mysql', 'sqlite'
        bucket_name='your-mlflow-artifacts-bucket'
    )
    
    # Alternative: Use local MLflow (comment out remote_config above)
    # mlflow.set_tracking_uri("http://localhost:5000")  # For local MLflow server
    
    # Configuration
    config = {
        "image_size": (224, 224),
        "batch_size": 32,
        "epochs": 10,
        "experiment_name": "defect-classification-experiments"
    }
    
    # Initialize classifier
    classifier = DefectClassifier(
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        experiment_name=config["experiment_name"]
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
        
        # Save the model
        classifier.save_model("defect_classifier_model.h5")
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved as 'defect_classifier_model.h5'")
        print(f"Check MLflow UI to view experiment results:")
        print(f"  mlflow ui")
        print(f"  Then navigate to http://localhost:5000")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise

def run_mlflow_ui():
    """Helper function to start MLflow UI"""
    import subprocess
    import sys
    
    try:
        print("Starting MLflow UI...")
        print("Navigate to http://localhost:5000 to view experiments")
        subprocess.run([sys.executable, "-m", "mlflow", "ui"])
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except Exception as e:
        print(f"Error starting MLflow UI: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "ui":
        run_mlflow_ui()
    else:
        main()
