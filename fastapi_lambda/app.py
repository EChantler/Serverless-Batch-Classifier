from fastapi import FastAPI
from mangum import Mangum

from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import boto3
import os
from botocore.exceptions import BotoCoreError, NoCredentialsError
import uuid 


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
    # S3 config - replace with your bucket name and region
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

    # Dummy classifier logic: always returns 'not defective'
    result = "not defective"
    return JSONResponse({
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
        responses.append({
            "customerId": item.customerId,
            "caseId": item.caseId,
            "classification": "pending",
            "imageUrl": s3_url
        })
    return JSONResponse(responses, status_code=status.HTTP_202_ACCEPTED)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
print("Hello, World!")