"""File upload endpoint for whitepapers and benchmark data."""
import os
import uuid
import re
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".json", ".txt", ".md", ".xlsx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and injection attacks."""
    # Strip any directory path components
    filename = os.path.basename(filename)
    # Remove any non-alphanumeric characters except dots, hyphens, and underscores
    name, ext = os.path.splitext(filename)
    name = re.sub(r'[^\w\-]', '_', name)
    # Prefix with a UUID to prevent collisions and enumeration
    return f"{uuid.uuid4().hex[:8]}_{name}{ext}"


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
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large. Maximum size is 50MB.")

    # Sanitize filename and save
    safe_name = _sanitize_filename(file.filename or "upload")
    filepath = os.path.join(UPLOAD_DIR, safe_name)

    # Verify the resolved path is within UPLOAD_DIR (defense in depth)
    real_upload_dir = os.path.realpath(UPLOAD_DIR)
    real_filepath = os.path.realpath(filepath)
    if not real_filepath.startswith(real_upload_dir):
        raise HTTPException(400, "Invalid filename.")

    with open(filepath, "wb") as f:
        f.write(contents)

    return {
        "filename": safe_name,
        "original_filename": os.path.basename(file.filename or ""),
        "size_bytes": len(contents),
        "status": "uploaded",
        "message": f"File uploaded successfully. It will be processed for model data extraction.",
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

