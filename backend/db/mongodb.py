"""MongoDB database connection and utilities."""
from typing import AsyncGenerator, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
import asyncio

from ..config import settings

logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB connection manager."""
    
    def __init__(self):
        """Initialize MongoDB connection settings."""
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self._connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """Build MongoDB connection string from settings."""
        if settings.MONGODB_CONNECTION_STRING:
            return settings.MONGODB_CONNECTION_STRING
        
        auth_part = ""
        if settings.MONGODB_USERNAME and settings.MONGODB_PASSWORD:
            auth_part = f"{settings.MONGODB_USERNAME}:{settings.MONGODB_PASSWORD}@"
        
        return f"mongodb://{auth_part}{settings.MONGODB_HOST}:{settings.MONGODB_PORT}/"
    
    async def connect(self, db_name: Optional[str] = None) -> None:
        """Connect to MongoDB and set up the database."""
        if self.client is not None:
            return
        
        try:
            # Set a shorter server selection timeout for faster failure
            self.client = AsyncIOMotorClient(
                self._connection_string,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=100,
                minPoolSize=10,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000,
                retryWrites=True,
                w="majority"
            )
            
            # Test the connection
            await self.client.admin.command('ping')
            
            # Set the database
            db_name = db_name or settings.MONGODB_DB
            self.db = self.client[db_name]
            
            logger.info("Successfully connected to MongoDB")
            
            # Initialize collections and indexes
            await self._initialize_collections()
            
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def _initialize_collections(self) -> None:
        """Initialize collections and create indexes."""
        if self.db is None:
            raise RuntimeError("Database not connected")
        
        # Import models to register their collections and indexes
        from ..models import models
        
        # Create indexes for all registered collections
        for model in [models.User, models.Prediction, models.Evaluation, models.APIKey]:
            if hasattr(model, 'Collection'):
                collection = model.Collection
                if hasattr(collection, 'indexes'):
                    for index_spec in collection.indexes:
                        if isinstance(index_spec, list):
                            # Simple index
                            keys = index_spec[0] if isinstance(index_spec[0], list) else [index_spec[0]]
                            index_options = index_spec[1] if len(index_spec) > 1 else {}
                            await self.db[collection.name].create_index(keys, **index_options)
                        else:
                            # Complex index (text, etc.)
                            await self.db[collection.name].create_index(index_spec)
        
        logger.info("Database indexes initialized")
    
    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB connection closed")
    
    async def get_database(self) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        if self.db is None:
            await self.connect()
        return self.db
    
    async def ping(self) -> bool:
        """Ping the MongoDB server to check if it's alive."""
        try:
            if self.client is None:
                await self.connect()
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False

# Global database manager instance
db_manager = MongoDBManager()

# Dependency to get the database in FastAPI routes
async def get_database() -> AsyncGenerator[AsyncIOMotorDatabase, None]:
    """Get the database connection for FastAPI dependency injection."""
    try:
        db = await db_manager.get_database()
        yield db
    except Exception as e:
        logger.error(f"Error getting database: {e}")
        raise

def get_sync_mongodb():
    """Get a synchronous MongoDB client for use in synchronous contexts."""
    return MongoClient(settings.MONGODB_CONNECTION_STRING or f"mongodb://{settings.MONGODB_HOST}:{settings.MONGODB_PORT}/")

# Test the connection
async def test_connection():
    """Test the MongoDB connection."""
    try:
        await db_manager.connect()
        return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False

# Run the test if this module is executed directly
if __name__ == "__main__":
    import asyncio
    
    async def main():
        connected = await test_connection()
        print(f"MongoDB connection test: {'SUCCESS' if connected else 'FAILED'}")
        if connected:
            # List all collections
            db = await db_manager.get_database()
            collections = await db.list_collection_names()
            print(f"Collections: {collections}")
            
            # Close the connection
            await db_manager.close()
    
    asyncio.run(main())
