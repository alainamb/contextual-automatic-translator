from fastapi import FastAPI
from app.database import connect_to_mongo
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Corpus-Graph Document Translation API")

@app.on_event("startup")
async def startup_event():
    mongo_uri = os.getenv("MONGODB_URI")
    await connect_to_mongo(mongo_uri)

# Include your routers here
from app.routers import documents
app.include_router(documents.router)