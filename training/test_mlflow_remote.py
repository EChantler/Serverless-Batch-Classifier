#!/usr/bin/env python3
"""
Test script for MLflow remote configuration
"""

import os
import sys
import mlflow
from mlflow_remote_config import MLflowRemoteConfig

def test_database_connection():
    """Test database connection"""
    print("Testing database connection...")
    
    try:
        config = MLflowRemoteConfig()
        db_type = os.getenv('DB_TYPE', 'postgresql')
        tracking_uri = config.get_database_uri(db_type)
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Try to create a test experiment
        experiment_name = "test-connection"
        try:
            experiment = mlflow.create_experiment(experiment_name)
            print(f"✓ Successfully created experiment: {experiment_name}")
            
            # Clean up
            mlflow.delete_experiment(experiment)
            print("✓ Successfully deleted test experiment")
            
        except Exception as e:
            if "already exists" in str(e):
                print(f"✓ Database connection working (experiment already exists)")
            else:
                raise
        
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_s3_connection():
    """Test S3 connection"""
    print("\nTesting S3 connection...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        config = MLflowRemoteConfig()
        bucket_name = os.getenv('MLFLOW_S3_BUCKET', 'mlflow-artifacts-bucket')
        
        s3_client = boto3.client('s3')
        
        # Test bucket access
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ Successfully accessed bucket: {bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"✗ Bucket not found: {bucket_name}")
                return False
            else:
                raise
        
        # Test write access
        test_key = "test-connection/test-file.txt"
        test_content = "MLflow S3 connection test"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content
        )
        print(f"✓ Successfully uploaded test file: {test_key}")
        
        # Clean up
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print("✓ Successfully deleted test file")
        
        return True
        
    except ImportError:
        print("✗ boto3 not installed. Install with: pip install boto3")
        return False
    except Exception as e:
        print(f"✗ S3 connection failed: {e}")
        return False

def test_mlflow_tracking():
    """Test MLflow tracking with remote backend"""
    print("\nTesting MLflow tracking...")
    
    try:
        config = MLflowRemoteConfig()
        tracking_uri, artifact_uri = config.configure_mlflow()
        
        # Create test experiment
        experiment_name = "remote-backend-test"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Start a test run
        with mlflow.start_run(experiment_id=experiment_id, run_name="test-run"):
            # Log some test metrics
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            
            # Test artifact logging
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("This is a test artifact")
                temp_file = f.name
            
            mlflow.log_artifact(temp_file, "test_artifacts")
            
            # Clean up temp file
            os.unlink(temp_file)
            
            print("✓ Successfully logged parameters, metrics, and artifacts")
        
        return True
        
    except Exception as e:
        print(f"✗ MLflow tracking failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MLflow Remote Configuration Test")
    print("=" * 40)
    
    # Load environment variables from .env file if it exists
    if os.path.exists('.env'):
        print("Loading environment variables from .env file...")
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Run tests
    tests = [
        test_database_connection,
        test_s3_connection,
        test_mlflow_tracking
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Database Connection: {'✓' if results[0] else '✗'}")
    print(f"S3 Connection: {'✓' if results[1] else '✗'}")
    print(f"MLflow Tracking: {'✓' if results[2] else '✗'}")
    
    if all(results):
        print("\n🎉 All tests passed! Your MLflow remote configuration is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check your configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
