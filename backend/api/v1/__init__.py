"""
API v1 package - Contains all version 1 API endpoints
"""
from fastapi import APIRouter

# Import all endpoint routers
from .endpoints import auth, predict, evaluate, admin

# Create the API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(predict.router, prefix="/predict", tags=["Predictions"])
api_router.include_router(evaluate.router, prefix="/evaluate", tags=["Model Evaluation"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
