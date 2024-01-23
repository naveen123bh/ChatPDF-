from fastapi import FastAPI, UploadFile, File, HTTPException, status
import os
import shutil

app = FastAPI()

UPLOAD_DIR = "data/pdf"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "ChatPDF API is running 🚀"}

# -------------------------
# Upload PDF endpoint
# -------------------------
@app.post("/upload-pdf")
async def upload_pdf(file:UploadFile= File(...)):
     # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status_code": status.HTTP_201_CREATED,
            "message": "File uploaded successfully",
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )