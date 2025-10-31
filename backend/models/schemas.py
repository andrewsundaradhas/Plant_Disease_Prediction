"""Pydantic models for request and response validation."""
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class DiseaseSeverity(str, Enum):
    HEALTHY = "healthy"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class PredictionConfidence(str, Enum):
    LOW = "low"        # 0-50%
    MEDIUM = "medium"   # 50-80%
    HIGH = "high"       # 80-95%
    VERY_HIGH = "very_high"  # 95-100%

# Base schemas
class BaseResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None
    
class ErrorResponse(BaseResponse):
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None

# User schemas
class UserBase(BaseModel):
    email: str = Field(..., example="user@example.com")
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, example="securepassword123")

class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class UserInDB(UserBase):
    id: str = Field(..., alias="_id")
    hashed_password: str
    
    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class UserResponse(BaseResponse):
    data: UserInDB

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None
    scopes: List[str] = []

# Prediction schemas
class PredictionBase(BaseModel):
    image_url: Optional[HttpUrl] = None
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = {}

class PredictionCreate(PredictionBase):
    user_id: Optional[str] = None

class PredictionInDB(PredictionBase):
    id: str = Field(..., alias="_id")
    user_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class PredictionResponse(BaseResponse):
    data: PredictionInDB

class PredictionListResponse(BaseResponse):
    data: List[PredictionInDB]
    total: int
    page: int
    limit: int
    total_pages: int

# Model evaluation schemas
class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    loss: Optional[float] = None
    class_metrics: Dict[str, Dict[str, float]]

class EvaluationRequest(BaseModel):
    test_data_dir: str
    batch_size: int = 32
    save_to_db: bool = True

class EvaluationResult(BaseModel):
    id: str = Field(..., alias="_id")
    metrics: EvaluationMetrics
    test_data_path: str
    model_version: str
    created_at: datetime
    created_by: Optional[str] = None
    
    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class EvaluationResponse(BaseResponse):
    data: EvaluationResult

# Health check schema
class HealthCheck(BaseModel):
    status: str = "ok"
    database: bool = False
    model_loaded: bool = False
    evaluation_in_progress: bool = False
    version: str
    timestamp: datetime

# File upload schemas
class FileUploadResponse(BaseResponse):
    file_id: str
    file_url: str
    file_path: str
    content_type: str
    file_size: int

# Batch prediction schemas
class BatchPredictionRequest(BaseModel):
    image_urls: List[HttpUrl]
    save_to_db: bool = True

class BatchPredictionItem(PredictionBase):
    image_url: HttpUrl
    prediction_id: str
    success: bool
    error: Optional[str] = None

class BatchPredictionResponse(BaseResponse):
    predictions: List[BatchPredictionItem]
    total: int
    successful: int
    failed: int
