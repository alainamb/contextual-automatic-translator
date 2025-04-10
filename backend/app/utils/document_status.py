from datetime import datetime
from bson import ObjectId
from app.database import get_db

def get_document_by_id(document_id):
    """Retrieve a document by its ID"""
    db = get_db()
    document = db.documents.find_one({"_id": ObjectId(document_id)})
    if not document:
        raise ValueError(f"Document with ID {document_id} not found")
    return document

def save_document(document):
    """Save updated document to database"""
    db = get_db()
    document["updated_at"] = datetime.utcnow()
    result = db.documents.update_one(
        {"_id": document["_id"]},
        {"$set": document}
    )
    return result.modified_count > 0

def update_document_status(document_id, new_status, changed_by, reason):
    """
    Update a document's status and maintain status change history
    
    Parameters:
    - document_id: The MongoDB ObjectId of the document
    - new_status: New status value ("new", "validated", "deprecated", "rejected")
    - changed_by: Username or ID of the person making the change
    - reason: Explanation for the status change
    
    Returns:
    - Boolean indicating success of the update operation
    """
    document = get_document_by_id(document_id)
    old_status = document.get("status")
    
    # Create the history entry
    history_entry = {
        "from_status": old_status,
        "to_status": new_status,
        "changed_at": datetime.utcnow(),
        "changed_by": changed_by,
        "reason": reason
    }
    
    # Update the document with new status
    document["status"] = new_status
    
    # Initialize history if it doesn't exist
    if "status_change_history" not in document:
        document["status_change_history"] = []
    
    # Add the new history entry
    document["status_change_history"].append(history_entry)
    
    # Add specific reason fields if applicable
    if new_status == "validated":
        document["validation_notes"] = reason
        document["validator_id"] = changed_by
    elif new_status == "rejected":
        document["rejection_reason"] = reason
        document["validator_id"] = changed_by
    elif new_status == "deprecated":
        document["deprecation_reason"] = reason
        document["validator_id"] = changed_by
    
    # Save the updated document
    return save_document(document)