""
ASGI config for Crop Health Prediction System.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""
import os
from app.main import app

# This allows uvicorn to find the FastAPI application
application = app

if __name__ == "__main__":
    import uvicorn
    
    # Run the application using uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
