"""
Prediction model for the Crop Health Prediction System.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from enum import Enum

# Custom Pydantic type for MongoDB ObjectId
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

# Prediction status enum
class PredictionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Base prediction model
class PredictionBase(BaseModel):
    """Base prediction model with common fields."""
    user_id: PyObjectId = Field(..., alias="user_id")
    image_path: str
    status: PredictionStatus = PredictionStatus.PENDING
    model_version: str = "1.0.0"
    metadata: Dict[str, Any] = {}

# Prediction creation model
class PredictionCreate(PredictionBase):
    """Model for creating a new prediction."""
    pass

# Prediction update model
class PredictionUpdate(BaseModel):
    """Model for updating a prediction."""
    status: Optional[PredictionStatus] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Prediction result model
class PredictionResult(BaseModel):
    """Model for prediction results."""
    predicted_class: str
    confidence: float
    all_predictions: Dict[str, float]
    inference_time: Optional[float] = None

# Prediction in database model
class PredictionInDB(PredictionBase):
    """Prediction model as stored in the database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        use_enum_values = True

# Prediction response model (sent to clients)
class Prediction(PredictionBase):
    """Prediction model for API responses."""
    id: str = Field(..., alias="_id")
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        use_enum_values = True

# Batch prediction model
class BatchPredictionCreate(BaseModel):
    """Model for creating a batch of predictions."""
    image_paths: List[str]
    model_version: str = "1.0.0"
    metadata: Dict[str, Any] = {}

# Batch prediction response model
class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    batch_id: str
    status: str
    created_at: datetime
    total_predictions: int
    completed_predictions: int = 0
    failed_predictions: int = 0
    prediction_ids: List[str] = []

# Prediction database operations
class PredictionDB:
    """Database operations for predictions."""
    
    @classmethod
    async def create(cls, db, prediction: PredictionCreate) -> PredictionInDB:
        """Create a new prediction in the database."""
        # Create prediction document
        prediction_dict = prediction.dict(by_alias=True)
        prediction_dict["created_at"] = datetime.utcnow()
        prediction_dict["updated_at"] = datetime.utcnow()
        
        # Insert into database
        result = await db.predictions.insert_one(prediction_dict)
        
        # Return the created prediction
        created_prediction = await db.predictions.find_one({"_id": result.inserted_id})
        return PredictionInDB(**created_prediction)
    
    @classmethod
    async def get_by_id(cls, db, prediction_id: str) -> Optional[PredictionInDB]:
        """Get a prediction by ID."""
        try:
            prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
            return PredictionInDB(**prediction) if prediction else None
        except:
            return None
    
    @classmethod
    async def get_by_user(
        cls, 
        db, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[PredictionStatus] = None
    ) -> List[PredictionInDB]:
        """Get predictions for a specific user."""
        query = {"user_id": ObjectId(user_id)}
        if status is not None:
            query["status"] = status
        
        predictions = []
        async for prediction in db.predictions.find(query).sort("created_at", -1).skip(skip).limit(limit):
            predictions.append(PredictionInDB(**prediction))
        
        return predictions
    
    @classmethod
    async def update(
        cls, 
        db, 
        prediction_id: str, 
        prediction_update: PredictionUpdate
    ) -> Optional[PredictionInDB]:
        """Update a prediction."""
        # Prepare update data
        update_data = prediction_update.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update prediction
        result = await db.predictions.update_one(
            {"_id": ObjectId(prediction_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            return None
        
        # Return updated prediction
        updated_prediction = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
        return PredictionInDB(**updated_prediction) if updated_prediction else None
    
    @classmethod
    async def delete(cls, db, prediction_id: str) -> bool:
        """Delete a prediction."""
        result = await db.predictions.delete_one({"_id": ObjectId(prediction_id)})
        return result.deleted_count > 0
    
    @classmethod
    async def list_predictions(
        cls, 
        db, 
        skip: int = 0, 
        limit: int = 100,
        user_id: Optional[str] = None,
        status: Optional[PredictionStatus] = None
    ) -> List[PredictionInDB]:
        """List predictions with pagination and filtering."""
        query = {}
        if user_id is not None:
            query["user_id"] = ObjectId(user_id)
        if status is not None:
            query["status"] = status
        
        predictions = []
        async for prediction in db.predictions.find(query).sort("created_at", -1).skip(skip).limit(limit):
            predictions.append(PredictionInDB(**prediction))
        
        return predictions
    
    @classmethod
    async def count(
        cls, 
        db, 
        user_id: Optional[str] = None, 
        status: Optional[PredictionStatus] = None
    ) -> int:
        """Count predictions matching a query."""
        query = {}
        if user_id is not None:
            query["user_id"] = ObjectId(user_id)
        if status is not None:
            query["status"] = status
            
        return await db.predictions.count_documents(query)
    
    @classmethod
    async def get_user_prediction_count(
        cls, 
        db, 
        user_id: str, 
        time_period: str = "day"
    ) -> int:
        """Get the number of predictions made by a user in a specific time period."""
        from datetime import datetime, timedelta
        
        # Calculate time delta based on period
        now = datetime.utcnow()
        if time_period == "day":
            start_time = now - timedelta(days=1)
        elif time_period == "week":
            start_time = now - timedelta(weeks=1)
        elif time_period == "month":
            start_time = now - timedelta(days=30)
        else:
            raise ValueError("Invalid time period. Must be 'day', 'week', or 'month'.")
        
        # Count predictions
        count = await db.predictions.count_documents({
            "user_id": ObjectId(user_id),
            "created_at": {"$gte": start_time}
        })
        
        return count
