from fastapi import FastAPI
from mangum import Mangum
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import boto3
import os
from botocore.exceptions import BotoCoreError, NoCredentialsError
import uuid
import pymysql

from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import datetime
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
import numpy as np
from PIL import Image
import tempfile

# Aurora MySQL connection string from env
DB_HOST = os.getenv("DB_HOST", "batch-classifier-db.ckx6ieea6u2k.us-east-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "MySecurePassword123!")
DB_NAME = os.getenv("DB_NAME", "batch_db")
S3_BUCKET = os.getenv("S3_BUCKET", "serverless-batch-classifier-bucket")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
# Ensure DB exists before SQLAlchemy engine creation
def ensure_database_exists():
    host = DB_HOST
    port = int(DB_PORT)
    user = DB_USER
    password = DB_PASSWORD
    db_name = DB_NAME
    try:
        conn = pymysql.connect(host=host, port=port, user=user, password=password)
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`;")
        conn.close()
    except Exception as e:
        print(f"Database creation error: {e}")

ensure_database_exists()

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Table model
class BatchRecord(Base):
    __tablename__ = "batch_records"
    uuid = Column(String(36), primary_key=True)
    customerId = Column(String(64))
    caseId = Column(String(64))
    imageUrl = Column(String(512))
    status = Column(String(32))
    modelVersion = Column(String(32))
    createdOn = Column(DateTime, default=datetime.datetime.utcnow)
    lastModifiedOn = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    createdBy = Column(String(64))
    lastModifiedBy = Column(String(64))

# Create table if not exists (for local dev)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"DB Table creation error: {e}")

## MLflow configuration (model loaded lazily)
# Set MLflow to use /tmp for local artifacts (writable in Lambda)
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", 
                                              f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/mlflow")
os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "/tmp/mlflow-artifacts"
os.environ["MLFLOW_ARTIFACT_ROOT"] = "/tmp/mlflow-artifacts"

# Prevent MLflow from creating ./mlruns directory
import tempfile
os.environ["TMPDIR"] = "/tmp"
os.environ["TMP"] = "/tmp"
os.environ["TEMP"] = "/tmp"

# Initialize MLflow with explicit tracking URI
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Global model cache
MODEL = None
MODEL_LOADED = False
MODEL_VERSION = None

def get_model():
    """Lazy load the model when first needed"""
    global MODEL, MODEL_LOADED, MODEL_VERSION
    
    if MODEL_LOADED and MODEL is not None:
        return MODEL, MODEL_VERSION
    
    try:
        import tensorflow as tf
        
        # Set additional environment variables to prevent MLflow from using local directories
        os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", 
                                                      f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/mlflow")
        
        model_name = os.getenv("MLFLOW_MODEL_NAME", "defect-classifier-model")
        model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
        
        print(f"Attempting to load model: {model_name} from stage: {model_stage}")
        
        # Try to get model version info first
        try:
            client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
            model_version_info = client.get_latest_versions(model_name, stages=[model_stage])
            if model_version_info:
                MODEL_VERSION = model_version_info[0].version
                model_uri = f"models:/{model_name}/{MODEL_VERSION}"
                print(f"Found model version: {MODEL_VERSION}")
            else:
                MODEL_VERSION = "unknown"
                model_uri = f"models:/{model_name}/{model_stage}"
                print(f"Could not find specific version, using stage: {model_stage}")
        except Exception as version_error:
            print(f"Could not get model version info: {version_error}")
            MODEL_VERSION = "unknown"
            model_uri = f"models:/{model_name}/{model_stage}"
        
        print(f"Loading model from URI: {model_uri}")
        
        # Create writable temp directory for MLflow
        temp_model_dir = f"/tmp/mlflow_model_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_model_dir, exist_ok=True)
        
        # Set MLflow to use our temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir("/tmp")  # Change to writable directory
            
            # Override MLflow's default artifact root
            import mlflow.store.artifact.utils.models
            
            loaded_model = mlflow.tensorflow.load_model(model_uri)
            MODEL = loaded_model
            MODEL_LOADED = True
            print(f"Successfully loaded model from registry: {model_uri}")
            return MODEL, MODEL_VERSION
            
        finally:
            os.chdir(original_cwd)  # Restore original directory
        
    except Exception as registry_error:
        print(f"Failed to load from model registry: {registry_error}")
        
        # Try alternative approach with direct S3 model loading if available
        try:
            print("Attempting alternative model loading approach...")
            
            # Check if we have a backup model in S3 or local fallback
            fallback_model_path = os.getenv("FALLBACK_MODEL_PATH")
            if fallback_model_path:
                print(f"Trying fallback model path: {fallback_model_path}")
                # This would need to be implemented based on your backup strategy
                # For now, we'll just set a basic fallback
                MODEL_VERSION = "fallback"
            else:
                MODEL_LOADED = True  # Mark as attempted to avoid repeated failures
                MODEL = None
                MODEL_VERSION = "failed"
                raise RuntimeError(f"Could not load model and no fallback available. Error: {registry_error}")
                
        except Exception as fallback_error:
            print(f"All loading methods failed: {fallback_error}")
            MODEL_LOADED = True  # Mark as attempted to avoid repeated failures
            MODEL = None
            MODEL_VERSION = "failed"
            raise RuntimeError(f"Could not load model. Registry error: {registry_error}, Fallback error: {fallback_error}")
    
    return MODEL, MODEL_VERSION

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}



# Classification endpoint
@app.post("/classify")
async def classify_image(
    image: UploadFile = File(...),
    customerId: str = Form(...),
    caseId: str = Form(...)
):
    
    s3_client = boto3.client("s3", region_name=S3_REGION)
    image_guid = uuid.uuid4()
    s3_key = f"uploads/{customerId}/{caseId}/image_{image_guid}.jpg"
    try:
        contents = await image.read()
        
        # Get model (lazy loading)
        model, model_version = get_model()
        
        if model is None:
            return JSONResponse({
                "error": "Model not loaded",
                "details": "Failed to load ML model for inference"
            }, status_code=500)
        
        # Perform inference
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp.flush()
            img = Image.open(tmp.name).resize((224, 224))
        x = np.array(img) / 255.0
        x = np.expand_dims(x, 0)
        # Run inference via MLflow loaded model
        predictions = model.predict(x)
        # extract prediction score
        score = predictions[0][0] if len(predictions.shape) > 1 else predictions[0]
        label = "not defective" if score > 0.5 else "defective"
        # Upload to S3
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=contents)
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    except RuntimeError as model_error:
        return JSONResponse({
            "error": "Model loading error",
            "details": str(model_error)
        }, status_code=500)
    except (BotoCoreError, NoCredentialsError) as e:
        return JSONResponse({
            "error": "Failed to upload image to S3 or inference error",
            "details": str(e)
        }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "error": "Unexpected error during classification",
            "details": str(e)
        }, status_code=500)

    # Insert into Aurora MySQL
    db = SessionLocal()
    record_uuid = str(uuid.uuid4())
    try:
        record = BatchRecord(
            uuid=record_uuid,
            customerId=customerId,
            caseId=caseId,
            imageUrl=s3_url,
            status=label,
            modelVersion=model_version,
            createdOn=datetime.datetime.utcnow(),
            lastModifiedOn=datetime.datetime.utcnow(),
            createdBy="api",
            lastModifiedBy="api"
        )
        db.add(record)
        db.commit()
    except SQLAlchemyError as db_err:
        db.close()
        return JSONResponse({
            "error": "Failed to write to DB",
            "details": str(db_err)
        }, status_code=500)
    db.close()

    return JSONResponse({
        "uuid": record_uuid,
        "customerId": customerId,
        "caseId": caseId,
        "classification": label,
        "score": float(score),
        "modelVersion": model_version,
        "imageUrl": s3_url
    })

# AWS Lambda handler
handler = Mangum(app)


# Batch classification endpoint


from fastapi import status
from typing import List
from pydantic import BaseModel
import base64


# Batch item metadata model

# Batch item metadata model
class BatchItem(BaseModel):
    image: str  # base64 encoded image
    customerId: str
    caseId: str



@app.post("/batch-classify")
async def batch_classify(batch: List[BatchItem]):
    """
    Accepts JSON body:
    [
      {"image": "<base64>", "customerId": "...", "caseId": "..."},
      ...
    ]
    """
    S3_BUCKET = os.getenv("S3_BUCKET", "serverless-batch-classifier-bucket")
    S3_REGION = os.getenv("S3_REGION", "us-east-1")
    s3_client = boto3.client("s3", region_name=S3_REGION)
    responses = []
    db = SessionLocal()
    for idx, item in enumerate(batch):
        try:
            image_bytes = base64.b64decode(item.image)
        except Exception as e:
            responses.append({
                "customerId": item.customerId,
                "caseId": item.caseId,
                "error": "Invalid base64 image",
                "details": str(e)
            })
            continue
        image_guid = uuid.uuid4()
        s3_key = f"uploads/{item.customerId}/{item.caseId}/{image_guid}.jpg"
        try:
            s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=image_bytes)
            s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        except (BotoCoreError, NoCredentialsError) as e:
            responses.append({
                "customerId": item.customerId,
                "caseId": item.caseId,
                "error": "Failed to upload image to S3",
                "details": str(e)
            })
            continue
        # Insert into Aurora MySQL
        record_uuid = str(uuid.uuid4())
        try:
            record = BatchRecord(
                uuid=record_uuid,
                customerId=item.customerId,
                caseId=item.caseId,
                imageUrl=s3_url,
                status="pending",
                createdOn=datetime.datetime.utcnow(),
                lastModifiedOn=datetime.datetime.utcnow(),
                createdBy="api",
                lastModifiedBy="api"
            )
            db.add(record)
            db.commit()
        except SQLAlchemyError as db_err:
            responses.append({
                "customerId": item.customerId,
                "caseId": item.caseId,
                "error": "Failed to write to DB",
                "details": str(db_err)
            })
            continue
        responses.append({
            "uuid": record_uuid,
            "customerId": item.customerId,
            "caseId": item.caseId,
            "classification": "pending",
            "imageUrl": s3_url
        })
    db.close()
    return JSONResponse(responses, status_code=status.HTTP_202_ACCEPTED)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
print("Hello, World!")