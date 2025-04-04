"""
Clustering models for document similarity - to be integrated with main models.py in the future

This file contains the data models needed to support document clustering functionality.
Once you've settled on your preferred clustering approach, these models can be integrated
into your main models.py file.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime
from bson import ObjectId

# Helper class for MongoDB ObjectIDs (from your existing models.py)
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

# Vector representation for documents
class DocumentVector(BaseModel):
    """Vector representation of a document or document section"""
    vector_type: str  # e.g., "sentence_transformer", "tfidf", "doc2vec"
    model_name: str  # e.g., "paraphrase-multilingual-MiniLM-L12-v2"
    dimensions: int  # e.g., 384, 768
    vector: List[float]  # The actual vector values
    document_id: PyObjectId
    section_id: Optional[str] = None  # If applicable
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Cluster metadata
class ClusterMetadata(BaseModel):
    """Information about how clusters were generated"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    algorithm: str  # "kmeans", "dbscan", "hierarchical", "lda", etc.
    parameters: Dict[str, Any]  # Algorithm-specific parameters
    feature_type: str  # Type of features used
    model_name: str  # Name of the model used for vectorization
    quality_metrics: Optional[Dict[str, float]] = None  # Metrics like silhouette score
    run_date: datetime = Field(default_factory=datetime.utcnow)
    run_by: Optional[str] = None  # User who ran the clustering
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Document cluster
class DocumentCluster(BaseModel):
    """A cluster of similar documents"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    cluster_id: str  # Unique identifier (e.g., "kmeans_3", "dbscan_1")
    name: Optional[str] = None  # Human-readable name
    description: Optional[str] = None  # Description of what this cluster represents
    size: int  # Number of documents
    centroid: Optional[List[float]] = None  # Vector representation of center (if applicable)
    keywords: Optional[List[str]] = None  # Keywords characterizing this cluster
    topics: Optional[Dict[str, float]] = None  # Topic distribution (if using topic modeling)
    parent_cluster_id: Optional[str] = None  # For hierarchical clustering
    level: Optional[str] = None  # For multi-level clustering
    created_at: datetime = Field(default_factory=datetime.utcnow)
    cluster_metadata_id: PyObjectId  # Reference to the clustering run
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Document membership in a cluster
class ClusterMembership(BaseModel):
    """A document's membership in a cluster"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    document_id: PyObjectId
    cluster_id: str
    membership_score: float = 1.0  # How strongly this document belongs (0-1)
    distance_to_centroid: Optional[float] = None  # Distance to cluster center
    is_core_point: Optional[bool] = None  # For DBSCAN
    is_noise: Optional[bool] = None  # For DBSCAN
    core_score: Optional[float] = None  # For DBSCAN
    rank_in_cluster: Optional[int] = None  # Position in the cluster
    level: Optional[str] = None  # For hierarchical clustering
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Similar document pair (traditional approach)
class DocumentSimilarity(BaseModel):
    """Traditional pairwise similarity between documents"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    document1_id: PyObjectId
    document2_id: PyObjectId
    similarity_score: float
    similarity_method: str  # e.g., "cosine", "jaccard"
    vector_type: str  # e.g., "document", "section"
    section1_id: Optional[str] = None
    section2_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    common_clusters: Optional[List[str]] = None  # Clusters both documents belong to
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Fields to add to DocumentMetadata
class ClusteringInfo(BaseModel):
    """Clustering information to add to DocumentMetadata"""
    primary_cluster_id: Optional[str] = None  # Main cluster this document belongs to
    cluster_ids: Optional[List[str]] = None  # All clusters this document belongs to
    clustering_info: Optional[Dict[str, Dict[str, Any]]] = None  # Algorithm-specific details
    # Example: {"kmeans": {"cluster_id": "kmeans_3", "distance": 0.1}, "dbscan": {...}}
    last_clustered_at: Optional[datetime] = None
    similar_documents: Optional[List[Dict[str, Any]]] = None  # Most similar documents
    # Example: [{"doc_id": "...", "score": 0.95, "common_clusters": ["c1", "c2"]}]
    vector_status: Optional[str] = None  # e.g., "generated", "outdated", "missing"


# When ready to integrate, add this to the DocumentMetadata class:
"""
# Add to DocumentMetadata class
clustering: Optional[ClusteringInfo] = None
"""