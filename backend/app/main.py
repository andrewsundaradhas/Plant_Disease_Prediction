""
Main FastAPI application module.
This module creates and configures the FastAPI application with all routes and middleware.
"""
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..core.config import settings
from ..db.mongodb import init_db, close_db
from ..api.v1 import api_router

# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database connection and create indexes
    await init_db()
    yield
    # Shutdown: Close database connection
    await close_db()

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Crop Health Prediction System API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Set up CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that provides API information."""
    return {
        "message": "Welcome to the Crop Health Prediction System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }
