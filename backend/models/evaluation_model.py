"""
Evaluation model for the Crop Health Prediction System.
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

# Evaluation status enum
class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Base evaluation model
class EvaluationBase(BaseModel):
    """Base evaluation model with common fields."""
    name: str
    description: Optional[str] = None
    model_version: str = "1.0.0"
    status: EvaluationStatus = EvaluationStatus.PENDING
    test_dataset_path: str
    metrics: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = {}

# Evaluation creation model
class EvaluationCreate(EvaluationBase):
    """Model for creating a new evaluation."""
    pass

# Evaluation update model
class EvaluationUpdate(BaseModel):
    """Model for updating an evaluation."""
    status: Optional[EvaluationStatus] = None
    metrics: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# Evaluation in database model
class EvaluationInDB(EvaluationBase):
    """Evaluation model as stored in the database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        use_enum_values = True

# Evaluation response model (sent to clients)
class Evaluation(EvaluationBase):
    """Evaluation model for API responses."""
    id: str = Field(..., alias="_id")
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        use_enum_values = True

# Evaluation database operations
class EvaluationDB:
    """Database operations for evaluations."""
    
    @classmethod
    async def create(cls, db, evaluation: EvaluationCreate) -> EvaluationInDB:
        """Create a new evaluation in the database."""
        # Create evaluation document
        evaluation_dict = evaluation.dict()
        evaluation_dict["created_at"] = datetime.utcnow()
        evaluation_dict["updated_at"] = datetime.utcnow()
        
        # Insert into database
        result = await db.evaluations.insert_one(evaluation_dict)
        
        # Return the created evaluation
        created_evaluation = await db.evaluations.find_one({"_id": result.inserted_id})
        return EvaluationInDB(**created_evaluation)
    
    @classmethod
    async def get_by_id(cls, db, evaluation_id: str) -> Optional[EvaluationInDB]:
        """Get an evaluation by ID."""
        try:
            evaluation = await db.evaluations.find_one({"_id": ObjectId(evaluation_id)})
            return EvaluationInDB(**evaluation) if evaluation else None
        except:
            return None
    
    @classmethod
    async def update(
        cls, 
        db, 
        evaluation_id: str, 
        evaluation_update: EvaluationUpdate
    ) -> Optional[EvaluationInDB]:
        """Update an evaluation."""
        # Prepare update data
        update_data = evaluation_update.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Handle status updates
        if "status" in update_data:
            if update_data["status"] == EvaluationStatus.RUNNING:
                update_data["started_at"] = datetime.utcnow()
            elif update_data["status"] in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
                update_data["completed_at"] = datetime.utcnow()
        
        # Update evaluation
        result = await db.evaluations.update_one(
            {"_id": ObjectId(evaluation_id)},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            return None
        
        # Return updated evaluation
        updated_evaluation = await db.evaluations.find_one({"_id": ObjectId(evaluation_id)})
        return EvaluationInDB(**updated_evaluation) if updated_evaluation else None
    
    @classmethod
    async def delete(cls, db, evaluation_id: str) -> bool:
        """Delete an evaluation."""
        result = await db.evaluations.delete_one({"_id": ObjectId(evaluation_id)})
        return result.deleted_count > 0
    
    @classmethod
    async def list_evaluations(
        cls, 
        db, 
        skip: int = 0, 
        limit: int = 100,
        status: Optional[EvaluationStatus] = None,
        model_version: Optional[str] = None
    ) -> List[EvaluationInDB]:
        """List evaluations with pagination and filtering."""
        query = {}
        if status is not None:
            query["status"] = status
        if model_version is not None:
            query["model_version"] = model_version
        
        evaluations = []
        async for evaluation in db.evaluations.find(query).sort("created_at", -1).skip(skip).limit(limit):
            evaluations.append(EvaluationInDB(**evaluation))
        
        return evaluations
    
    @classmethod
    async def count(
        cls, 
        db, 
        status: Optional[EvaluationStatus] = None,
        model_version: Optional[str] = None
    ) -> int:
        """Count evaluations matching a query."""
        query = {}
        if status is not None:
            query["status"] = status
        if model_version is not None:
            query["model_version"] = model_version
            
        return await db.evaluations.count_documents(query)
    
    @classmethod
    async def get_latest_evaluation(
        cls, 
        db, 
        model_version: Optional[str] = None,
        status: EvaluationStatus = EvaluationStatus.COMPLETED
    ) -> Optional[EvaluationInDB]:
        """Get the most recent evaluation."""
        query = {"status": status}
        if model_version is not None:
            query["model_version"] = model_version
        
        evaluation = await db.evaluations.find_one(
            query,
            sort=[("completed_at", -1)]
        )
        
        return EvaluationInDB(**evaluation) if evaluation else None
    
    @classmethod
    async def get_model_metrics(
        cls, 
        db, 
        model_version: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get metrics for a specific model version."""
        pipeline = [
            {"$match": {
                "model_version": model_version,
                "status": EvaluationStatus.COMPLETED,
                "metrics": {"$exists": True}
            }},
            {"$sort": {"completed_at": -1}},
            {"$limit": limit},
            {"$project": {
                "_id": 0,
                "evaluation_id": "$_id",
                "completed_at": 1,
                "metrics": 1
            }}
        ]
        
        metrics = []
        async for doc in db.evaluations.aggregate(pipeline):
            metrics.append({
                "evaluation_id": str(doc["evaluation_id"]),
                "completed_at": doc["completed_at"],
                "metrics": doc["metrics"]
            })
        
        return metrics
