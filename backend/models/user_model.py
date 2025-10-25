"""
User model for the Crop Health Prediction System.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, EmailStr, Field, validator
from bson import ObjectId
import bcrypt
import re

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

# Base user model
class UserBase(BaseModel):
    """Base user model with common fields."""
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

# User creation model
class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str

    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

# User update model
class UserUpdate(BaseModel):
    """Model for updating user information."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None

    @validator('password')
    def password_strength(cls, v):
        if v is not None:
            if len(v) < 8:
                raise ValueError('Password must be at least 8 characters long')
            if not re.search(r"[A-Z]", v):
                raise ValueError('Password must contain at least one uppercase letter')
            if not re.search(r"[a-z]", v):
                raise ValueError('Password must contain at least one lowercase letter')
            if not re.search(r"\d", v):
                raise ValueError('Password must contain at least one number')
        return v

# User in database model
class UserInDB(UserBase):
    """User model as stored in the database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

# User response model (sent to clients)
class User(UserBase):
    """User model for API responses."""
    id: str = Field(..., alias="_id")
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True

# User authentication model
class UserLogin(BaseModel):
    """Model for user login."""
    email: EmailStr
    password: str

# Token model
class Token(BaseModel):
    """Authentication token model."""
    access_token: str
    token_type: str = "bearer"

# Token data model
class TokenData(BaseModel):
    """Data stored in the authentication token."""
    user_id: Optional[str] = None
    email: Optional[EmailStr] = None
    is_superuser: bool = False

# Password reset request model
class PasswordResetRequest(BaseModel):
    """Model for requesting a password reset."""
    email: EmailStr

# Password reset model
class PasswordReset(BaseModel):
    """Model for resetting a password."""
    token: str
    new_password: str

    @validator('new_password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"[a-z]", v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

# User database operations
class UserDB:
    """Database operations for users."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password for storing in the database."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a stored password against one provided by user."""
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    
    @classmethod
    async def create(cls, db, user: UserCreate) -> UserInDB:
        """Create a new user in the database."""
        # Check if user with this email already exists
        existing_user = await db.users.find_one({"email": user.email})
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Hash the password
        hashed_password = cls.hash_password(user.password)
        
        # Create user document
        user_dict = user.dict(exclude={"password"})
        user_dict["hashed_password"] = hashed_password
        user_dict["created_at"] = datetime.utcnow()
        user_dict["updated_at"] = datetime.utcnow()
        
        # Insert into database
        result = await db.users.insert_one(user_dict)
        
        # Return the created user
        created_user = await db.users.find_one({"_id": result.inserted_id})
        return UserInDB(**created_user)
    
    @classmethod
    async def authenticate(cls, db, email: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user."""
        # Find user by email
        user = await db.users.find_one({"email": email})
        if not user:
            return None
        
        # Verify password
        if not cls.verify_password(password, user["hashed_password"]):
            return None
        
        # Update last login time
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        return UserInDB(**user)
    
    @classmethod
    async def get_by_id(cls, db, user_id: str) -> Optional[UserInDB]:
        """Get a user by ID."""
        try:
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            return UserInDB(**user) if user else None
        except:
            return None
    
    @classmethod
    async def get_by_email(cls, db, email: str) -> Optional[UserInDB]:
        """Get a user by email."""
        user = await db.users.find_one({"email": email})
        return UserInDB(**user) if user else None
    
    @classmethod
    async def update(
        cls, 
        db, 
        user_id: str, 
        user_update: Dict[str, Any]
    ) -> Optional[UserInDB]:
        """Update a user."""
        # Handle password update
        if "password" in user_update:
            user_update["hashed_password"] = cls.hash_password(user_update.pop("password"))
        
        # Update user
        user_update["updated_at"] = datetime.utcnow()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": user_update}
        )
        
        if result.matched_count == 0:
            return None
        
        # Return updated user
        updated_user = await db.users.find_one({"_id": ObjectId(user_id)})
        return UserInDB(**updated_user) if updated_user else None
    
    @classmethod
    async def delete(cls, db, user_id: str) -> bool:
        """Delete a user."""
        result = await db.users.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0
    
    @classmethod
    async def list_users(
        cls, 
        db, 
        skip: int = 0, 
        limit: int = 100,
        is_active: Optional[bool] = None
    ) -> List[UserInDB]:
        """List users with pagination and filtering."""
        query = {}
        if is_active is not None:
            query["is_active"] = is_active
        
        users = []
        async for user in db.users.find(query).skip(skip).limit(limit):
            users.append(UserInDB(**user))
        
        return users
    
    @classmethod
    async def count(cls, db, query: Optional[Dict[str, Any]] = None) -> int:
        """Count users matching a query."""
        return await db.users.count_documents(query or {})
