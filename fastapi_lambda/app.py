
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

# Aurora MySQL connection string from env
DB_HOST = os.getenv("DB_HOST", "batch-classifier-db.ckx6ieea6u2k.us-east-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "MySecurePassword123!")
DB_NAME = os.getenv("DB_NAME", "batch_db")

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
    createdOn = Column(DateTime, default=datetime.datetime.utcnow)
    lastModifiedOn = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    createdBy = Column(String(64))
    lastModifiedBy = Column(String(64))

# Create table if not exists (for local dev)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"DB Table creation error: {e}")


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
    S3_BUCKET = os.getenv("S3_BUCKET", "serverless-batch-classifier-bucket")
    S3_REGION = os.getenv("S3_REGION", "us-east-1")
    s3_client = boto3.client("s3", region_name=S3_REGION)
    image_guid = uuid.uuid4()
    s3_key = f"uploads/{customerId}/{caseId}/image_{image_guid}.jpg"
    try:
        contents = await image.read()
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=contents)
        s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    except (BotoCoreError, NoCredentialsError) as e:
        return JSONResponse({
            "error": "Failed to upload image to S3",
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
            status="not defective",
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

    result = "not defective"
    return JSONResponse({
        "uuid": record_uuid,
        "customerId": customerId,
        "caseId": caseId,
        "classification": result,
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