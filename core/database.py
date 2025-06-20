import os
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGODB_URI = os.getenv(
    "MONGODB_URI", 
    "mongodb://localhost:27017/"
)
DATABASE_NAME = os.getenv("DATABASE_NAME", "comp430_benchmark")
CONNECTION_TIMEOUT = int(os.getenv("MONGODB_TIMEOUT", "5000"))  # 5 seconds

# Global variables for database connection
client = None
db = None
questions_collection = None
responses_collection = None

def initialize_database():
    """Initialize database connection with proper error handling."""
    global client, db, questions_collection, responses_collection
    
    try:
        logger.info(f"🔄 Attempting to connect to MongoDB at: {MONGODB_URI}")
        
        # Create MongoDB client with timeout settings
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=CONNECTION_TIMEOUT,
            connectTimeoutMS=CONNECTION_TIMEOUT,
            socketTimeoutMS=CONNECTION_TIMEOUT
        )
        
        # Test the connection
        client.admin.command('ping')
        
        # Initialize database and collections
        db = client[DATABASE_NAME]
        questions_collection = db["questions"]
        responses_collection = db["responses"]
        
        logger.info("✅ MongoDB connection established successfully!")
        
        # Ensure question versioning only if connection is successful
        ensure_question_versioning()
        
        return True
        
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        logger.warning(f"⚠️  MongoDB connection failed: {str(e)}")
        logger.warning("📱 App will run in demo mode without database functionality")
        
        # Set collections to None to indicate no database connection
        client = None
        db = None
        questions_collection = None
        responses_collection = None
        
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected database error: {str(e)}")
        client = None
        db = None
        questions_collection = None
        responses_collection = None
        return False

def ensure_question_versioning():
    """Ensure all questions have original_q_id field for version tracking."""
    if questions_collection is None:
        logger.warning("⚠️  Skipping question versioning - no database connection")
        return
    
    try:
        # Find all questions without original_q_id
        questions_without_version = questions_collection.find({"original_q_id": {"$exists": False}})
        
        updated_count = 0
        for question in questions_without_version:
            # For version 1 questions, set original_q_id to their q_id
            if question.get("q_version") == "1" or "q_version" not in question:
                questions_collection.update_one(
                    {"_id": question["_id"]},
                    {"$set": {"original_q_id": question["q_id"], "q_version": "1"}}
                )
                updated_count += 1
        
        if updated_count > 0:
            logger.info(f"📝 Updated {updated_count} questions with version tracking")
        else:
            logger.info("✅ All questions already have version tracking")
            
    except Exception as e:
        logger.error(f"❌ Error during question versioning: {str(e)}")

def is_database_connected():
    """Check if database is connected and available."""
    return client is not None and db is not None

def get_database_status():
    """Get database connection status for display."""
    if is_database_connected():
        return {
            "status": "connected",
            "uri": MONGODB_URI,
            "database": DATABASE_NAME,
            "message": "Database connection active"
        }
    else:
        return {
            "status": "disconnected",
            "uri": MONGODB_URI,
            "database": DATABASE_NAME,
            "message": "Running in demo mode - database not available"
        }

# Initialize database connection on import
logger.info("📦 Initializing MongoDB connection...")
database_connected = initialize_database()

if database_connected:
    logger.info("🚀 Database initialization complete!")
else:
    logger.info("🚀 App starting in demo mode without database!") 