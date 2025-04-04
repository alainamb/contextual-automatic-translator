from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection string from environment variables
mongo_uri = os.getenv("MONGODB_URI")

# Create MongoDB client (this doesn't connect yet, it just configures the client)
client = MongoClient(mongo_uri)

# Access specific database
db = client.translation_corpus # Update to match database name

# Export collection references to use in other parts of the application
documents = db.documents #Update to match what's on MongoDB
metadata = db.metadata #Update to match what's on MongoDB
document_pairs = db.document_pairs #Update to match what's on MongoDB
processed_texts = db.processed_texts #Update to match what's on MongoDB
translation_jobs = db.translation_jobs #Update to match what's on MongoDB

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING

db = None

async def get_database():
    return db

async def connect_to_mongo(mongo_uri):
    global db
    client = AsyncIOMotorClient(mongo_uri)
    db = client.corpus_translation_db
    
    # Create indexes
    await create_indexes()
    
    return db

async def create_indexes():
    # Create indexes for faster querying
    document_metadata = db.document_metadata
    
    indexes = [
        IndexModel([("title", ASCENDING)], background=True),
        IndexModel([("document_type", ASCENDING)], background=True),
        IndexModel([("keywords", ASCENDING)], background=True),
        IndexModel([("language_variant", ASCENDING)], background=True),
        IndexModel([("category", ASCENDING)], background=True),
        IndexModel([("status", ASCENDING)], background=True),
        # Add more indexes as needed for your query patterns
    ]
    
    await document_metadata.create_indexes(indexes)