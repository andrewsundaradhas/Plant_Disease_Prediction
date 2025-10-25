"""Database models for the Crop Health Prediction API."""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, HttpUrl
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

# Custom ObjectId type for Pydantic
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

# Base model with common fields
class BaseDBModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# User model
class User(BaseDBModel):
    """User model for authentication and authorization."""
    email: str = Field(..., unique=True)
    hashed_password: str
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    last_login: Optional[datetime] = None
    
    class Collection:
        name = "users"
        indexes = [
            [("email", 1), {"unique": True}],
            [("is_active", 1)],
        ]
    
    def to_response(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

# Prediction model
class Prediction(BaseDBModel):
    """Prediction model for storing prediction results."""
    user_id: Optional[PyObjectId] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: Dict[str, float]
    metadata: Dict[str, Any] = {}
    
    class Collection:
        name = "predictions"
        indexes = [
            [("user_id", 1)],
            [("predicted_class", 1)],
            [("confidence", 1)],
            [("created_at", -1)],
        ]
    
    @classmethod
    async def create(
        cls, 
        db: AsyncIOMotorDatabase, 
        prediction_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> 'Prediction':
        """Create a new prediction record in the database."""
        if user_id:
            prediction_data["user_id"] = ObjectId(user_id)
        
        # Ensure class probabilities are floats
        if "class_probabilities" in prediction_data:
            prediction_data["class_probabilities"] = {
                k: float(v) for k, v in prediction_data["class_probabilities"].items()
            }
        
        result = await db[cls.Collection.name].insert_one(prediction_data)
        return await cls.get(db, str(result.inserted_id))
    
    @classmethod
    async def get(cls, db: AsyncIOMotorDatabase, prediction_id: str) -> Optional['Prediction']:
        """Get a prediction by ID."""
        prediction = await db[cls.Collection.name].find_one({"_id": ObjectId(prediction_id)})
        if prediction:
            return cls(**prediction)
        return None
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "image_url": self.image_url,
            "image_path": self.image_path,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

# Evaluation model
class Evaluation(BaseDBModel):
    """Model for storing model evaluation results."""
    metrics: Dict[str, Any]
    test_data_path: str
    model_version: str
    created_by: Optional[PyObjectId] = None
    
    class Collection:
        name = "evaluations"
        indexes = [
            [("model_version", 1)],
            [("created_at", -1)],
            [("created_by", 1)],
        ]
    
    @classmethod
    async def create(
        cls, 
        db: AsyncIOMotorDatabase, 
        metrics: Dict[str, Any],
        test_data_path: str,
        model_version: str,
        user_id: Optional[str] = None
    ) -> 'Evaluation':
        """Create a new evaluation record in the database."""
        evaluation_data = {
            "metrics": metrics,
            "test_data_path": test_data_path,
            "model_version": model_version,
        }
        
        if user_id:
            evaluation_data["created_by"] = ObjectId(user_id)
        
        result = await db[cls.Collection.name].insert_one(evaluation_data)
        return await cls.get(db, str(result.inserted_id))
    
    @classmethod
    async def get_latest(cls, db: AsyncIOMotorDatabase, model_version: Optional[str] = None) -> Optional['Evaluation']:
        """Get the most recent evaluation, optionally filtered by model version."""
        query = {}
        if model_version:
            query["model_version"] = model_version
        
        evaluation = await db[cls.Collection.name].find_one(
            query,
            sort=[("created_at", -1)]
        )
        
        if evaluation:
            return cls(**evaluation)
        return None
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": str(self.id),
            "metrics": self.metrics,
            "test_data_path": self.test_data_path,
            "model_version": self.model_version,
            "created_by": str(self.created_by) if self.created_by else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

# API Key model for authentication
class APIKey(BaseDBModel):
    """API Key model for programmatic access."""
    name: str
    key: str
    user_id: PyObjectId
    expires_at: Optional[datetime] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    
    class Collection:
        name = "api_keys"
        indexes = [
            [("key", 1), {"unique": True}],
            [("user_id", 1)],
            [("is_active", 1)],
        ]
    
    @classmethod
    async def get_by_key(cls, db: AsyncIOMotorDatabase, key: str) -> Optional['APIKey']:
        """Get an API key by its value."""
        api_key = await db[cls.Collection.name].find_one({"key": key, "is_active": True})
        if api_key:
            return cls(**api_key)
        return None
    
    def is_expired(self) -> bool:
        """Check if the API key has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "id": str(self.id),
            "name": self.name,
            "key": self.key,
            "user_id": str(self.user_id),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
