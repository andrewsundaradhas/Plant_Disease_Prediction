""
Authentication and user management endpoints.

This module provides endpoints for user registration, login, and token management.
"""
from datetime import timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.errors import DuplicateKeyError

from ....core.security import (
    create_access_token,
    get_password_hash,
    get_current_user,
    get_current_active_superuser,
    get_user,
    authenticate_user,
    oauth2_scheme,
    get_api_key_user,
)
from ....core.config import settings
from ....models.schemas import (
    Token,
    UserInDB,
    UserCreate,
    UserResponse,
    ErrorResponse,
    BaseResponse,
)
from ....models.models import User, APIKey
from ....db.mongodb import db_manager

router = APIRouter()

# Token expiration times
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


@router.post("/login", response_model=Token, responses={
    400: {"model": ErrorResponse, "description": "Incorrect username or password"}
})
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Dict[str, Any]:
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    - **username**: The user's email
    - **password**: The user's password
    
    Returns an access token that can be used for authenticated requests.
    """
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Determine scopes based on user role
    scopes = ["user", "predict"]  # Default scopes
    if user.is_superuser:
        scopes.extend(["admin", "evaluate"])
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "scopes": scopes},
        expires_delta=access_token_expires,
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds()),
        "user": {
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
        }
    }


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_in: UserCreate,
    current_user: User = Depends(get_current_active_superuser)
) -> Dict[str, Any]:
    """
    Register a new user.
    
    - **email**: The user's email (must be unique)
    - **password**: The user's password (at least 8 characters)
    - **full_name**: The user's full name (optional)
    - **is_active**: Whether the user is active (admin only, defaults to True)
    - **is_superuser**: Whether the user is a superuser (admin only, defaults to False)
    
    Returns the created user without the hashed password.
    """
    db = await db_manager.get_database()
    
    # Check if user already exists
    existing_user = await get_user(user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_in.password)
    user_data = user_in.dict(exclude={"password"})
    user_data["hashed_password"] = hashed_password
    
    try:
        # Insert the new user
        result = await db["users"].insert_one(user_data)
        user_data["_id"] = result.inserted_id
        
        # Return the created user (without the hashed password)
        return {
            "success": True,
            "data": {
                "id": str(user_data["_id"]),
                "email": user_data["email"],
                "full_name": user_data.get("full_name"),
                "is_active": user_data.get("is_active", True),
                "is_superuser": user_data.get("is_superuser", False),
            }
        }
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get the current user's information.
    
    Returns the authenticated user's details.
    """
    return {
        "success": True,
        "data": {
            "id": str(current_user.id),
            "email": current_user.email,
            "full_name": current_user.full_name,
            "is_active": current_user.is_active,
            "is_superuser": current_user.is_superuser,
        }
    }


@router.post("/api-keys", response_model=Dict[str, Any])
async def create_api_key(
    name: str = Body(..., embed=True),
    expires_in_days: Optional[int] = Body(30, embed=True, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Create a new API key for the current user.
    
    - **name**: A name for the API key (for identification)
    - **expires_in_days**: Number of days until the key expires (1-365, default: 30)
    
    Returns the new API key (only shown once).
    """
    import secrets
    from datetime import datetime, timedelta
    
    # Generate a secure random API key
    api_key = f"ck_{secrets.token_urlsafe(32)}"
    
    # Calculate expiration date
    expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
    
    # Store the API key in the database
    db = await db_manager.get_database()
    result = await db["api_keys"].insert_one({
        "name": name,
        "key": api_key,
        "user_id": current_user.id,
        "expires_at": expires_at,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    })
    
    return {
        "success": True,
        "data": {
            "id": str(result.inserted_id),
            "name": name,
            "key": api_key,  # Only shown once!
            "expires_at": expires_at.isoformat(),
        },
        "message": "API key created successfully. Save this key as it won't be shown again."
    }


@router.get("/api-keys", response_model=Dict[str, Any])
async def list_api_keys(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    List all API keys for the current user.
    
    Returns a list of API keys (without the actual key values for security).
    """
    db = await db_manager.get_database()
    cursor = db["api_keys"].find({"user_id": current_user.id})
    
    keys = []
    async for key in cursor:
        keys.append({
            "id": str(key["_id"]),
            "name": key["name"],
            "expires_at": key["expires_at"].isoformat() if key.get("expires_at") else None,
            "is_active": key.get("is_active", True),
            "last_used": key.get("last_used").isoformat() if key.get("last_used") else None,
            "created_at": key["created_at"].isoformat(),
        })
    
    return {
        "success": True,
        "data": keys,
        "count": len(keys)
    }


@router.delete("/api-keys/{key_id}", response_model=BaseResponse)
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Revoke an API key.
    
    - **key_id**: The ID of the API key to revoke
    
    Returns a success message if the key was revoked.
    """
    from bson import ObjectId
    
    db = await db_manager.get_database()
    result = await db["api_keys"].update_one(
        {"_id": ObjectId(key_id), "user_id": current_user.id},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or access denied"
        )
    
    return {
        "success": True,
        "message": "API key revoked successfully"
    }
