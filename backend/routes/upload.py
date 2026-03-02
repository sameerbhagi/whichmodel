"""File upload endpoint for whitepapers and benchmark data."""
import os
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".txt", ".md", ".xlsx"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a whitepaper, benchmark CSV, or other model data file."""
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"File type '{ext}' not allowed. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Validate size (max 50MB)
    contents = await file.read()
    max_size = 50 * 1024 * 1024
    if len(contents) > max_size:
        raise HTTPException(400, "File too large. Maximum size is 50MB.")

    # Save file
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    return {
        "filename": file.filename,
        "size_bytes": len(contents),
        "status": "uploaded",
        "message": f"File '{file.filename}' uploaded successfully. It will be processed for model data extraction.",
    }


@router.get("/uploads")
def list_uploads():
    """List all uploaded files."""
    files = []
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath):
                files.append({
                    "filename": fname,
                    "size_bytes": os.path.getsize(fpath),
                })
    return {"files": files, "count": len(files)}
