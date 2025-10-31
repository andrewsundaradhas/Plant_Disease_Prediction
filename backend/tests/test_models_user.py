""
Tests for the User model and related functionality.
"""
import pytest
from bson import ObjectId
from datetime import datetime, timedelta

# Import test utilities
from tests.test_utils import assert_error_response

@pytest.mark.asyncio
async def test_create_user(db):
    """Test creating a new user."""
    from models.user_model import UserDB, UserCreate, UserInDB
    
    # Create test user data
    user_data = {
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "TestPass123!",
        "is_active": True,
        "is_superuser": False
    }
    
    # Create user
    user = await UserDB.create(db, UserCreate(**user_data))
    
    # Verify user was created
    assert user is not None
    assert user.email == user_data["email"]
    assert user.full_name == user_data["full_name"]
    assert user.is_active == user_data["is_active"]
    assert user.is_superuser == user_data["is_superuser"]
    assert hasattr(user, "hashed_password")
    assert user.hashed_password != user_data["password"]  # Password should be hashed
    assert hasattr(user, "created_at")
    assert hasattr(user, "updated_at")
    assert user.created_at <= datetime.utcnow()
    assert user.updated_at <= datetime.utcnow()

@pytest.mark.asyncio
async def test_duplicate_email(db):
    """Test that duplicate email addresses are not allowed."""
    from models.user_model import UserDB, UserCreate
    
    # Create first user
    user_data = {
        "email": "duplicate@example.com",
        "full_name": "First User",
        "password": "TestPass123!"
    }
    await UserDB.create(db, UserCreate(**user_data))
    
    # Try to create user with same email
    with pytest.raises(ValueError, match="User with this email already exists"):
        await UserDB.create(db, UserCreate(**user_data))

@pytest.mark.asyncio
async def test_authenticate_user(db):
    """Test user authentication."""
    from models.user_model import UserDB, UserCreate
    
    # Create test user
    password = "TestPass123!"
    user_data = {
        "email": "auth@example.com",
        "full_name": "Auth User",
        "password": password
    }
    user = await UserDB.create(db, UserCreate(**user_data))
    
    # Test successful authentication
    authenticated_user = await UserDB.authenticate(
        db, 
        email=user_data["email"], 
        password=password
    )
    assert authenticated_user is not None
    assert authenticated_user.email == user_data["email"]
    
    # Test failed authentication (wrong password)
    failed_auth = await UserDB.authenticate(
        db,
        email=user_data["email"],
        password="wrongpassword"
    )
    assert failed_auth is None
    
    # Test non-existent user
    non_existent = await UserDB.authenticate(
        db,
        email="nonexistent@example.com",
        password="anypassword"
    )
    assert non_existent is None

@pytest.mark.asyncio
async def test_get_user_by_id(db, test_user):
    """Test retrieving a user by ID."""
    from models.user_model import UserDB
    
    # Get the test user
    user = await UserDB.get_by_id(db, test_user["id"])
    
    # Verify user data
    assert user is not None
    assert str(user.id) == test_user["id"]
    assert user.email == test_user["email"]
    assert user.full_name == test_user["full_name"]

@pytest.mark.asyncio
async def test_get_nonexistent_user(db):
    """Test retrieving a non-existent user."""
    from models.user_model import UserDB
    
    # Try to get a user with a non-existent ID
    non_existent_id = str(ObjectId())
    user = await UserDB.get_by_id(db, non_existent_id)
    assert user is None

@pytest.mark.asyncio
async def test_update_user(db, test_user):
    """Test updating a user's information."""
    from models.user_model import UserDB, UserUpdate
    
    # Update user data
    update_data = {
        "full_name": "Updated Name",
        "is_active": False
    }
    
    # Perform update
    updated_user = await UserDB.update(
        db, 
        user_id=test_user["id"],
        user_update=UserUpdate(**update_data)
    )
    
    # Verify updates
    assert updated_user is not None
    assert updated_user.full_name == update_data["full_name"]
    assert updated_user.is_active == update_data["is_active"]
    assert updated_user.updated_at > datetime.utcnow() - timedelta(seconds=5)
    
    # Verify other fields remain unchanged
    assert updated_user.email == test_user["email"]

@pytest.mark.asyncio
async def test_update_user_password(db, test_user):
    """Test updating a user's password."""
    from models.user_model import UserDB, UserUpdate, UserInDB
    
    # Get the current user
    user = await UserDB.get_by_id(db, test_user["id"])
    old_hashed_password = user.hashed_password
    
    # Update password
    new_password = "NewSecurePass123!"
    update_data = {
        "password": new_password
    }
    
    # Perform update
    await UserDB.update(
        db,
        user_id=test_user["id"],
        user_update=UserUpdate(**update_data)
    )
    
    # Get updated user
    updated_user = await UserDB.get_by_id(db, test_user["id"])
    
    # Verify password was updated
    assert updated_user.hashed_password != old_hashed_password
    
    # Verify we can authenticate with new password
    authenticated = await UserDB.authenticate(
        db,
        email=test_user["email"],
        password=new_password
    )
    assert authenticated is not None

@pytest.mark.asyncio
async def test_delete_user(db, test_user):
    """Test deleting a user."""
    from models.user_model import UserDB
    
    # Delete the user
    deleted = await UserDB.delete(db, test_user["id"])
    assert deleted is True
    
    # Verify user no longer exists
    user = await UserDB.get_by_id(db, test_user["id"])
    assert user is None

@pytest.mark.asyncio
async def test_list_users(db, test_user, test_superuser):
    """Test listing users with pagination."""
    from models.user_model import UserDB
    
    # Get all users
    users = await UserDB.list_users(db)
    
    # Should find both test users
    assert len(users) >= 2
    user_emails = [user.email for user in users]
    assert test_user["email"] in user_emails
    assert test_superuser["email"] in user_emails
    
    # Test pagination
    users_page1 = await UserDB.list_users(db, skip=0, limit=1)
    assert len(users_page1) == 1
    
    users_page2 = await UserDB.list_users(db, skip=1, limit=1)
    assert len(users_page2) == 1
    
    # Verify different pages return different users
    if users_page1[0].email == users_page2[0].email:
        # If emails are the same, it's the same user, which shouldn't happen with proper pagination
        assert False, "Pagination returned the same user on different pages"

@pytest.mark.asyncio
async def test_count_users(db, test_user, test_superuser):
    """Test counting users."""
    from models.user_model import UserDB
    
    # Count all users
    count = await UserDB.count(db)
    assert count >= 2  # At least the test user and superuser
    
    # Count active users
    active_count = await UserDB.count(db, query={"is_active": True})
    assert active_count >= 2  # Both test users are active
    
    # Count superusers
    superuser_count = await UserDB.count(db, query={"is_superuser": True})
    assert superuser_count >= 1  # At least the test superuser

@pytest.mark.asyncio
async def test_password_hashing():
    """Test password hashing and verification."""
    from models.user_model import UserDB
    
    password = "TestPass123!"
    hashed = UserDB.hash_password(password)
    
    # Verify the same password hashes differently each time
    assert hashed != UserDB.hash_password(password)
    
    # Verify correct password
    assert UserDB.verify_password(password, hashed) is True
    
    # Verify incorrect password
    assert UserDB.verify_password("wrongpassword", hashed) is False
