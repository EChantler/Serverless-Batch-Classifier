import os
import mlflow
import boto3
import botocore.exceptions

class MLflowRemoteConfig:
    def configure_mlflow(
        self,
        db_type,
        bucket_name,
        region_name,
        db_host=None,
        db_port=None,
        db_user=None,
        db_password=None,
        db_name="mlflow"
    ):
        """
        Configure MLflow to use a remote backend store (MySQL or SQLite)
        and S3 artifact store. Returns (tracking_uri, artifact_uri).
        """
        # Set AWS region for S3 artifact store
        os.environ.setdefault("AWS_DEFAULT_REGION", region_name)

        # Build backend store URI
        if db_type == 'mysql':
            try:
                import pymysql
            except ImportError:
                raise ImportError(
                    "pymysql is required for MySQL backend. Install it with 'pip install pymysql'"
                )
            backend_store_uri = (
                f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
        elif db_type == 'sqlite':
            db_path = os.path.join(os.getcwd(), f"{db_name}.db")
            backend_store_uri = f"sqlite:///{db_path}"
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

        # Ensure S3 bucket exists or create it
        s3 = boto3.client('s3', region_name=region_name)
        try:
            s3.head_bucket(Bucket=bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response.get('Error', {}).get('Code', 0))
            # If bucket does not exist, create it
            # if error_code == 404:
            #     create_kwargs = {'Bucket': bucket_name}
            #     # Specify location for non-default regions
            #     if region_name != 'us-east-1':
            #         create_kwargs['CreateBucketConfiguration'] = {'LocationConstraint': region_name}
            #     s3.create_bucket(**create_kwargs)
            # else:
            raise
        # Build artifact store URI
        artifact_uri = f"s3://{bucket_name}/{db_name}"

        # Configure MLflow
        mlflow.set_tracking_uri(backend_store_uri)

        return backend_store_uri, artifact_uri
