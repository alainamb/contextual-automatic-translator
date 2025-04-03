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