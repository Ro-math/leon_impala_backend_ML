import os
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter()

LOGS_DIR = Path("data/logs")

@router.delete("/", status_code=200)
def delete_all_logs():
    """
    Deletes all files in the data/logs directory.
    """
    if not LOGS_DIR.exists():
        return {"message": "Logs directory does not exist, nothing to delete."}
    
    deleted_count = 0
    errors = []

    try:
        for file_path in LOGS_DIR.iterdir():
            if file_path.is_file():
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"Failed to delete {file_path.name}: {str(e)}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error accessing logs directory: {str(e)}")

    if errors:
        return {
            "message": f"Deleted {deleted_count} logs with some errors.",
            "errors": errors
        }
    
    return {"message": f"Successfully deleted {deleted_count} log files."}
