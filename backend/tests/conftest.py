""
Pytest configuration and fixtures for testing the Crop Health Prediction System.
"""
import os
import asyncio
import pytest
from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
import pytest_asyncio

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["MONGODB_URL"] = "mongodb://localhost:27017/test_crop_health"
os.environ["JWT_SECRET_KEY"] = "test_secret_key"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_S3_BUCKET_NAME"] = "test-bucket"
os.environ["AWS_DYNAMODB_TABLE_NAME"] = "test-table"
os.environ["AWS_SQS_QUEUE_URL"] = "https://sqs.test.amazonaws.com/test-queue"

# Import after setting environment variables
from app.main import app
from core.database import get_db
from core.config import settings

@pytest.fixture(scope="session")
def event_loop():
    ""
    Create an instance of the default event loop for the test session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="function")
async def client():
    ""
    Create a test client for the FastAPI application.
    """
    from fastapi.testclient import TestClient
    
    async def override_get_db():
        # Use the test database
        test_db = AsyncIOMotorClient(settings.MONGODB_URL)["test_db"]
        try:
            yield test_db
        finally:
            # Clean up after the test
            await test_db.client.drop_database("test_db")
    
    # Override the database dependency
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clear overrides
    app.dependency_overrides.clear()

@pytest_asyncio.fixture(scope="function")
async def db():
    ""
    Provide a database connection for testing.
    """
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client["test_db"]
    
    # Create indexes
    await db.users.create_index("email", unique=True)
    await db.predictions.create_index("user_id")
    await db.predictions.create_index("status")
    await db.evaluations.create_index("status")
    
    try:
        yield db
    finally:
        # Clean up after the test
        await client.drop_database("test_db")
        client.close()

@pytest_asyncio.fixture(scope="function")
async def test_user(db):
    ""
    Create a test user and return user data and credentials.
    """
    from models.user_model import UserDB
    
    user_data = {
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "TestPass123!",
        "is_active": True,
        "is_superuser": False
    }
    
    # Create the user
    user = await UserDB.create(db, user_data)
    
    # Return user data and credentials
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "credentials": {
            "username": user.email,
            "password": user_data["password"]
        }
    }

@pytest_asyncio.fixture(scope="function")
async def test_superuser(db):
    ""
    Create a test superuser and return user data and credentials.
    """
    from models.user_model import UserDB
    
    user_data = {
        "email": "admin@example.com",
        "full_name": "Admin User",
        "password": "AdminPass123!",
        "is_active": True,
        "is_superuser": True
    }
    
    # Create the superuser
    user = await UserDB.create(db, user_data)
    
    # Return user data and credentials
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "credentials": {
            "username": user.email,
            "password": user_data["password"]
        }
    }

@pytest_asyncio.fixture(scope="function")
async def auth_headers(client, test_user):
    ""
    Get authentication headers for the test user.
    """
    # Log in the test user
    login_data = {
        "username": test_user["email"],
        "password": test_user["credentials"]["password"]
    }
    
    response = client.post(
        "/api/v1/auth/login",
        data=login_data
    )
    
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}

@pytest_asyncio.fixture(scope="function")
async def admin_auth_headers(client, test_superuser):
    ""
    Get authentication headers for the test superuser.
    """
    # Log in the test superuser
    login_data = {
        "username": test_superuser["email"],
        "password": test_superuser["credentials"]["password"]
    }
    
    response = client.post(
        "/api/v1/auth/login",
        data=login_data
    )
    
    assert response.status_code == 200
    token = response.json()["access_token"]
    
    return {"Authorization": f"Bearer {token}"}

@pytest_asyncio.fixture(scope="function")
async def test_prediction(db, test_user):
    ""
    Create a test prediction.
    """
    from models.prediction_model import PredictionDB
    
    prediction_data = {
        "user_id": ObjectId(test_user["id"]),
        "image_path": "test/path/to/image.jpg",
        "status": "pending",
        "model_version": "1.0.0",
        "metadata": {"test": True}
    }
    
    # Create the prediction
    prediction = await db.predictions.insert_one(prediction_data)
    prediction_data["_id"] = prediction.inserted_id
    
    return prediction_data

@pytest_asyncio.fixture(scope="function")
async def test_evaluation(db):
    ""
    Create a test evaluation.
    """
    from models.evaluation_model import EvaluationDB
    
    evaluation_data = {
        "name": "Test Evaluation",
        "description": "Test evaluation description",
        "model_version": "1.0.0",
        "status": "pending",
        "test_dataset_path": "test/path/to/dataset",
        "metadata": {"test": True}
    }
    
    # Create the evaluation
    evaluation = await db.evaluations.insert_one(evaluation_data)
    evaluation_data["_id"] = evaluation.inserted_id
    
    return evaluation_data

# Mock AWS services for testing
@pytest.fixture(scope="function")
def mock_s3(monkeypatch):
    ""
    Mock the S3 service for testing.
    """
    from unittest.mock import AsyncMock, MagicMock
    
    mock_client = MagicMock()
    mock_client.upload_file = AsyncMock(return_value=None)
    mock_client.download_file = AsyncMock(return_value=None)
    mock_client.generate_presigned_url = MagicMock(return_value="https://test-presigned-url.com")
    mock_client.delete_object = AsyncMock(return_value={"ResponseMetadata": {"HTTPStatusCode": 204}})
    
    mock_resource = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.upload_fileobj = AsyncMock(return_value=None)
    mock_bucket.download_fileobj = AsyncMock(return_value=None)
    mock_resource.Bucket.return_value = mock_bucket
    
    monkeypatch.setattr("boto3.client", MagicMock(return_value=mock_client))
    monkeypatch.setattr("boto3.resource", MagicMock(return_value=mock_resource))
    
    return {"client": mock_client, "resource": mock_resource}

@pytest.fixture(scope="function")
def mock_dynamodb(monkeypatch):
    ""
    Mock the DynamoDB service for testing.
    """
    from unittest.mock import MagicMock, AsyncMock
    
    mock_client = MagicMock()
    mock_client.put_item = AsyncMock(return_value={"ResponseMetadata": {"HTTPStatusCode": 200}})
    mock_client.get_item = AsyncMock(return_value={"Item": {}})
    mock_client.update_item = AsyncMock(return_value={"Attributes": {}})
    mock_client.delete_item = AsyncMock(return_value={"ResponseMetadata": {"HTTPStatusCode": 200}})
    mock_client.query = AsyncMock(return_value={"Items": [], "Count": 0})
    mock_client.scan = AsyncMock(return_value={"Items": [], "Count": 0})
    
    monkeypatch.setattr("boto3.client", MagicMock(return_value=mock_client))
    
    return mock_client

@pytest.fixture(scope="function")
def mock_sqs(monkeypatch):
    ""
    Mock the SQS service for testing.
    """
    from unittest.mock import MagicMock, AsyncMock
    
    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(return_value={"MessageId": "test-message-id"})
    mock_client.receive_message = AsyncMock(return_value={"Messages": []})
    mock_client.delete_message = AsyncMock(return_value={"ResponseMetadata": {"HTTPStatusCode": 200}})
    mock_client.change_message_visibility = AsyncMock(return_value=None)
    
    monkeypatch.setattr("boto3.client", MagicMock(return_value=mock_client))
    
    return mock_client
