"""Security utilities for the Crop Health Prediction API."""
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from pydantic import ValidationError

from ..config import settings
from ..models.schemas import TokenData, UserInDB
from ..models.models import User, APIKey
from ..db.mongodb import db_manager

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/token",
    scopes={
        "user": "Read information about the current user.",
        "predict": "Make predictions using the model.",
        "evaluate": "Evaluate model performance.",
        "admin": "Admin access.",
    }
)

# JWT token functions
def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a new JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        bool: True if the password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)

async def get_user(email: str) -> Optional[User]:
    """Get a user by email.
    
    Args:
        email: User email
        
    Returns:
        User object if found, None otherwise
    """
    db = await db_manager.get_database()
    user_data = await db["users"].find_one({"email": email, "is_active": True})
    if user_data:
        return User(**user_data)
    return None

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate a user.
    
    Args:
        email: User email
        password: Plain text password
        
    Returns:
        User object if authentication succeeds, None otherwise
    """
    user = await get_user(email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    """Get the current authenticated user from the JWT token.
    
    Args:
        security_scopes: Security scopes required for the endpoint
        token: JWT token from the Authorization header
        
    Returns:
        Authenticated user
        
    Raises:
        HTTPException: If authentication fails or scopes are insufficient
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope=\"{security_scopes.scope_str}\"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False}
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(scopes=token_scopes, email=email)
    except (JWTError, ValidationError):
        raise credentials_exception
    
    user = await get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    
    # Check scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            if "admin" in token_data.scopes:
                # Admins have all scopes
                continue
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active user
        
    Raises:
        HTTPException: If the user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active superuser.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active superuser
        
    Raises:
        HTTPException: If the user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="The user doesn't have enough privileges"
        )
    return current_user

# API Key authentication
async def get_api_key(api_key: str) -> Optional[APIKey]:
    """Get an API key from the database.
    
    Args:
        api_key: API key string
        
    Returns:
        APIKey object if found and active, None otherwise
    """
    db = await db_manager.get_database()
    key_data = await db["api_keys"].find_one({"key": api_key, "is_active": True})
    if key_data:
        return APIKey(**key_data)
    return None

async def get_api_key_user(api_key: str = Depends(oauth2_scheme)) -> User:
    """Get the user associated with an API key.
    
    Args:
        api_key: API key string
        
    Returns:
        User associated with the API key
        
    Raises:
        HTTPException: If the API key is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate API key",
        headers={"WWW-Authenticate": "APIKey"},
    )
    
    try:
        key = await get_api_key(api_key)
        if key is None or key.is_expired():
            raise credentials_exception
        
        # Update last used timestamp
        db = await db_manager.get_database()
        await db["api_keys"].update_one(
            {"_id": key.id},
            {"$set": {"last_used": datetime.utcnow()}}
        )
        
        # Get the user
        user = await get_user_by_id(str(key.user_id))
        if not user:
            raise credentials_exception
            
        return user
        
    except Exception as e:
        raise credentials_exception

async def get_user_by_id(user_id: str) -> Optional[User]:
    """Get a user by ID.
    
    Args:
        user_id: User ID
        
    Returns:
        User object if found, None otherwise
    """
    from bson import ObjectId
    
    db = await db_manager.get_database()
    user_data = await db["users"].find_one({"_id": ObjectId(user_id), "is_active": True})
    if user_data:
        return User(**user_data)
    return None

# Rate limiting
class RateLimiter:
    """Simple rate limiter using in-memory storage."""
    
    def __init__(self, requests: int, window: int):
        """Initialize the rate limiter.
        
        Args:
            requests: Number of requests allowed in the time window
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.timestamps = []
    
    async def is_allowed(self) -> bool:
        """Check if a request is allowed.
        
        Returns:
            bool: True if the request is allowed, False otherwise
        """
        now = time.time()
        
        # Remove timestamps older than the time window
        self.timestamps = [t for t in self.timestamps if now - t < self.window]
        
        if len(self.timestamps) < self.requests:
            self.timestamps.append(now)
            return True
        
        return False
