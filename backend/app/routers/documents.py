from fastapi import APIRouter, HTTPException, Depends, Body
from app.utils.document_status import update_document_status
from app.models import DocumentMetadata
from typing import Dict

router = APIRouter()

@router.put("/documents/{document_id}/status", response_model=DocumentMetadata)
async def update_status(
    document_id: str,
    status_update: Dict = Body(...),
    # Add authentication dependency here when implemented
):
    """
    Update the status of a document
    
    Body should contain:
    - new_status: The new status value
    - reason: Explanation for the status change
    """
    if "new_status" not in status_update or "reason" not in status_update:
        raise HTTPException(status_code=400, detail="Missing required fields")
        
    # In a real implementation, you'd get the user from the auth token
    # For now we'll hardcode or pass it in the request
    changed_by = status_update.get("changed_by", "system")
    
    try:
        success = update_document_status(
            document_id,
            status_update["new_status"],
            changed_by,
            status_update["reason"]
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update document status")
            
        # Return the updated document
        # You'd need to implement a function to retrieve the document
        updated_document = get_document_by_id(document_id)
        return updated_document
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))