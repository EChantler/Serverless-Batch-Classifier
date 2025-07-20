import base64
import io
from fastapi.testclient import TestClient
from fastapi_lambda.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_classify_image(monkeypatch):
    # Mock S3 upload
    monkeypatch.setattr("boto3.client", lambda *a, **kw: type("S3", (), {"put_object": lambda self, **kw: None})())
    img_bytes = b"testimage"
    file = io.BytesIO(img_bytes)
    file.name = "test.jpg"
    response = client.post(
        "/classify",
        files={"image": (file.name, file, "image/jpeg")},
        data={"customerId": "123", "caseId": "A"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["customerId"] == "123"
    assert data["caseId"] == "A"
    assert data["classification"] == "not defective"
    assert "imageUrl" in data

def test_batch_classify(monkeypatch):
    # Mock S3 upload
    monkeypatch.setattr("boto3.client", lambda *a, **kw: type("S3", (), {"put_object": lambda self, **kw: None})())
    img_bytes = b"testimage"
    b64_img = base64.b64encode(img_bytes).decode()
    batch = [
        {"image": b64_img, "customerId": "123", "caseId": "A"},
        {"image": b64_img, "customerId": "456", "caseId": "B"}
    ]
    response = client.post("/batch-classify", json=batch)
    assert response.status_code == 202
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert item["classification"] == "pending"
        assert "imageUrl" in item
        assert "customerId" in item
        assert "caseId" in item
