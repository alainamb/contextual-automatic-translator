from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from bson import ObjectId
from typing import Optional, List, Dict, Any, Union
from datetime import datetime

# Helper for working with MongoDB ObjectIDs in Pydantic
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

# Vector and Section models
class DocumentVector(BaseModel):
    """Represents a vector embedding for a document or section"""
    vector_type: str  # "document" or "section"
    vector: List[float]  # The actual vector embedding
    model_used: str  # e.g., "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    section_id: Optional[str] = None  # Only for section vectors, to identify which section
    dimensions: int  # Dimensionality of the vector (e.g., 384, 768)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True

class DocumentSection(BaseModel):
    """Represents a section of a document"""
    section_id: str  # Unique identifier for the section within the document
    title: Optional[str] = None
    content: str
    start_offset: int  # Character offset in the full document
    end_offset: int
    section_type: Optional[str] = None  # e.g., "introduction", "methodology", "conclusion"
    vectors: Optional[List[DocumentVector]] = None
    
    class Config:
        arbitrary_types_allowed = True

# Author models

# LANGUAGE VARIATION AND AUTHOR LOCATION DATA
# We collect detailed information about authors' countries of origin, nationality, 
# and residence history to account for linguistic nuance in our corpus.
# Rationale:
# 1. Authors often exhibit hybrid language patterns influenced by their migration history
#    (e.g., a Venezuelan who has lived in Mexico for 10 years may write in a 
#    hybrid Venezuelan/Mexican Spanish variant)
# 2. This data helps us better understand dialectal variations within language families
# 3. It allows for more accurate document attribution and context-sensitive translation
# 4. It supports research on how geography influences language patterns and terminology
# 5. This information is crucial for training translation models sensitive to regional variations

class AuthorDemographics(BaseModel):
    """Demographic information about an author"""
    author_country_of_origin: Optional[str] = None  # The country where the author was born/raised
    current_country_of_residence: Optional[str] = None  # Where the author currently lives
    previous_countries: Optional[List[Dict[str, Union[str, int]]]] = None  # Previous residences
    nationalities: Optional[List[str]] = None
    races: Optional[List[str]] = None
    ethnicities: Optional[List[str]] = None
    gender: Optional[str] = None  # Options include "non-binary", "transgender female", "transgender male", "cisgender female", "cisgender male"
    sexual_orientation: Optional[str] = None

class PersonAuthor(BaseModel):
    """Information about an individual person as an author"""
    paternal_last_name: str
    maternal_last_name: Optional[str] = None
    first_name: str
    demographics: Optional[AuthorDemographics] = None
    is_primary: bool = False  # Indicates if this is the primary author
    is_translator: bool = False  # Indicates if the author is the translator of a work
    # As a general rule, translations should not be included in a great number in a corpus
    
    class Config:
        arbitrary_types_allowed = True

class OrganizationAuthor(BaseModel):
    """Information about an organization as an author"""
    institution_name: str
    institution_type: Optional[str] = None
    organization_country_of_origin: Optional[str] = None  # Where the organization was founded
    current_headquarters_country: Optional[str] = None  # Current headquarters location
    is_primary: bool = False  # Indicates if this is the primary author
    is_translator: bool = False  # Indicates if the author is the translator of a work
    # As a general rule, translations should not be included in a great number in a corpus

# Document similarity tracking
class DocumentSimilarity(BaseModel):
    source_doc_id: PyObjectId # MongoDB ObjectId 
    target_doc_id: PyObjectId # MongoDB ObjectId 
    source_content_hash: Optional[str] = None  # REVISION Content-based hash of source document - Source needs to be updated here so as to not confuse this word with how "source" is used in the field of translation; below     content_hash: str  # Your application's content-based hash identifier ; can "content_hash" just be used in two places? Here source_content_hash refers to the document in question that is being recorded no?
    target_content_hash: Optional[str] = None  # REVISION Content-based hash of target document - Target needs to be updated here so as to not confuse this word with how "target" is used in the field of translation
    similarity_score: float
    vector_type: str  # "document" or "section"
    section_pair: Optional[Dict[str, str]] = None  # Maps source section_id to target section_id
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Main Document Metadata model
class DocumentMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")  # MongoDB's internal ID
    content_hash: str  # Your application's content-based hash identifier
    original_filename: str  # The original filename as uploaded by the user
    title: str
    authorship_type: str  # "person_authored", "organization_authored", or "mixed_authorship"
    document_country_of_origin: Optional[str] = None  # Where the document was initially created/written
    
    # Shared fields
    ## Status tracking
    permission: str # "publicly available", "permission obtained from primary author", "permission obtained from publisher", "permission not obtained - for private collection"
    status: str  # "new", "validated", "deprecated", "rejected"
    status_change_history: List[Dict[str, Union[str, datetime]]] = Field(
        default_factory=list
    )  # Track status changes with timestamps and reasons
    rejection_reason: Optional[str] = None  # Explanation if rejected
    deprecation_reason: Optional[str] = None  # Explanation if deprecated
    validation_notes: Optional[str] = None  # Notes from validation process
    validator_id: Optional[str] = None  # Person who performed validation

    ## Document routing information
    corpora: str # "tl (translation & localization)", "gai", "usmex (U.S.-Mexico relations)"
    language_family: str  # ISO 639-3 (e.g., "spa", "eng")
    language_variant: str # ISO 3166-1 alpha-3 (e.g., "MEX", "USA") - Content in English not from the USA routed to the en-intl corpus folder; Content from countries in Latin America where Spanish is the main language routed to LATAM corpus, all other Spanish content is routed to es-intl folder

    ## Content Classification and Discovery Metadata
    category: Optional[str] = None  # Initially empty - to be populated by clustering algorithms later
    keywords: List[str] = [] # Can be generated or manually provided - Let's build in functionality to generate this automatically
    summary: Optional[str] = None # Aids in AI processing, searching, and human understanding - Let's build in functionality to generate this automatically

    ## Specifications - matches translation specifications in UI
    text_type: str 
    purpose: str 
    point_of_view: str
    audience: str
    reach: str
    word_count: str # Approximate value calculated by GAI from txt files once those have been created; used to measure the reliability of the corpus in terms of word count - for special language corpora, a starter word count of 100,000 is recommended

    ## Book info if applicable
    book_title: Optional[str] = None
    editors: Optional[List[str]] = []

    ## Journal info if applicable
    journal_title: Optional[str] = None

    ## Magazine info if applicable
    magazine_title: Optional[str] = None

    ## Publication info
    publication_year: Optional[int] = None
    publisher: Optional[str] = None

    ## Citation information (supporting multiple standards)
    # Place of publication and page range retained for comprehensive citation support
    place_of_publication: Optional[str] = None
    page_range_start: Optional[int] = None
    page_range_end: Optional[int] = None
    citation_formats: Optional[Dict[str, str]] = None  # Can store pre-formatted citations in different styles
    
    # Content storage
    # QUESTION Why were all of these marked as optional?
    original_file_path: str  # Path to original unprocessed document
    processed_file_path: str  # Path to processed text file for vector generation
    corpus_path: str  # e.g., "tl/en-US/processed/Smith_TranslationQuality_2022.txt" - Isn't this a redundant item with "processed_file_path"? Is this needed?
    content_text: Optional[str] = None  # The full text content
    # Optional for performance reasons with large documents
    content_url: Optional[str] = None  # Optional, only for linked rather than uploaded content

    # Document structure
    # QUESTION How is this part of the file supposed to work?
    sections: Optional[List[DocumentSection]] = None # Sections of the document, required even if just one section
    
    # Vector embeddings
    document_vector: Optional[DocumentVector] = None  # Vector for the entire document
    section_vectors: Optional[List[DocumentVector]] = None  # Vectors for individual sections
    
    # For knowledge graph connections
    # TODO: Expand entity_references when implementing the knowledge graph component
    entity_references: Optional[Dict[str, List[str]]] = None  # Entity type -> list of entity IDs
    
    # Author information - organizations or people or both
    person_authors: Optional[List[PersonAuthor]] = None
    organization_authors: Optional[List[OrganizationAuthor]] = None
    
    # Metadata about the submission itself
    # QUESTION Wouldn't it be logical for this information to be presented above the content storage section?
    contributor: str  # Person submitting this content
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None  # Additional information provided by the contributor
    
    # REVISION This example part needs to be updated once we've finalized the schema above
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "title": "Translation Quality Metrics",
                "authorship_type": "person_authored",
                "status": "validated",
                "permission": "publicly_available",
                "language_family": "eng",
                "language_variant": "English - United States",
                "country_code": "USA",
                "subcorpora": ["Quality Management"],
                "category": "Translation",
                "subcategory": "quality_assessment",
                "keywords": ["metrics", "quality", "assessment"],
                "summary": "This paper discusses translation quality metrics...",
                "text_type": "academic",
                "audience": ["Quality Managers", "Researchers/Academics"],
                "word_count": 3500,
                "file_path": "Smith_TranslationQuality_2022.txt",
                "original_file_path": "Smith_TranslationQuality_2022.pdf",
                "corpus_path": "tl/en-US/processed/Smith_TranslationQuality_2022.txt",
                "content_url": None,
                "publication_year": 2022,
                "publisher": "Journal of Translation Studies",
                "place_of_publication": "New York",
                "book_journal_title": "Journal of Translation Studies",
                "editors": [],
                "translators": [],
                "person_authors": [
                    {
                        "paternal_last_name": "Smith",
                        "maternal_last_name": "",
                        "first_name": "John",
                        "is_primary": True,
                        "demographics": {
                            "nationality": "United States",
                            "race": None,
                            "ethnicity": None,
                            "gender": "Male",
                            "sexual_orientation": None
                        }
                    },
                    {
                        "paternal_last_name": "Garcia",
                        "maternal_last_name": "",
                        "first_name": "Maria",
                        "is_primary": False,
                        "demographics": {
                            "nationality": "Mexico",
                            "race": None,
                            "ethnicity": None,
                            "gender": "Female",
                            "sexual_orientation": None
                        }
                    }
                ],
                "organization_authors": [],
                "contributor": "Jane Doe",
                "created_at": "2025-04-01T12:00:00.000Z",
                "updated_at": "2025-04-01T12:00:00.000Z",
                "sections": [
                    {
                        "section_id": "introduction",
                        "title": "Introduction",
                        "content": "This paper explores...",
                        "start_offset": 0,
                        "end_offset": 500,
                        "section_type": "introduction"
                    }
                ],
                "vectors": [
                    {
                        "vector_type": "document",
                        "vector": [0.1, 0.2, 0.3, 0.4],
                        "model_used": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        "dimensions": 384,
                        "created_at": "2025-04-01T12:00:00.000Z"
                    }
                ],
                "entity_references": {
                    "standard": ["ISO_17100", "ASTM_F2575"],
                    "concept": ["translation_quality", "metrics"]
                }
            }
        }

    schema_extra = {
    "example": {
        # Your other example fields...
        "status": "validated",
        "status_change_history": [
            {
                "from_status": None,
                "to_status": "new",
                "changed_at": "2025-04-01T10:00:00.000Z",
                "changed_by": "submission_system", 
                "reason": "Initial submission"
            },
            {
                "from_status": "new",
                "to_status": "validated", 
                "changed_at": "2025-04-02T14:30:00.000Z",
                "changed_by": "validator_username",
                "reason": "Document meets all corpus inclusion criteria"
            }
        ],
        # Continue with other example fields...
    }
}

# For response models (without certain fields)
# QUESTION What is this part for?
class DocumentMetadataResponse(BaseModel):
    id: str = Field(..., alias="_id")
    title: str
    authorship_type: str
    status: str
    language_family: str
    category: Optional[str] = None
    keywords: List[str] = []
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}