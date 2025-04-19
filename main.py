from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import os
from dotenv import load_dotenv
import models
import schemas
from models import TeaserStatus
from database import get_db, engine
from parser.pdf_parser import PDFParser
from parser.nlp import NLPProcessor
from pipeline.teaser_pipeline import TeaserProcessingPipeline

# Load environment variables
load_dotenv()

# Create tables in the database
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Teaser AI API",
    description="API for parsing and analyzing private equity teasers",
    version="1.0.0",
)

# Print OpenAI API key availability for debugging (without revealing the key itself)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    print(f"OpenAI API key is available in main.py. First 5 chars: {openai_api_key[:5]}...")
else:
    print("WARNING: OpenAI API key is NOT available in main.py. GPT analysis will be skipped.")

# Configure CORS with settings from environment variables
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NLP processor
nlp_processor = NLPProcessor()

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Background task for processing PDFs
async def process_pdf(file_content: bytes, filename: str, db: Session):
    # Extract text from PDF
    extracted_text = await PDFParser.extract_text_from_pdf(file_content)
    
    # Update teaser with extracted text
    db_teaser = db.query(models.Teaser).filter(models.Teaser.filename == filename).order_by(models.Teaser.id.desc()).first()
    
    if db_teaser:
        db_teaser.extracted_text = extracted_text
        db_teaser.status = TeaserStatus.PROCESSING
        db.commit()
        db.refresh(db_teaser)
        
        # Start the pipeline processing
        pipeline = TeaserProcessingPipeline(db, nlp_processor)
        await pipeline.process(db_teaser.id)

@app.post("/upload", response_model=schemas.TeaserResponse)
async def upload_teaser(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF teaser file for parsing and analysis.
    """
    # Check if the uploaded file is a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read file content
    file_content = await file.read()
    
    # Create a teaser in the database with processing status
    db_teaser = models.Teaser(
        filename=file.filename,
        status=TeaserStatus.PROCESSING
    )
    db.add(db_teaser)
    db.commit()
    db.refresh(db_teaser)
    
    # Process the PDF in the background
    background_tasks.add_task(process_pdf, file_content, file.filename, db)
    
    return db_teaser

@app.get("/teasers", response_model=schemas.TeaserList)
async def get_teasers(db: Session = Depends(get_db)):
    """
    Get all teasers from the database.
    """
    teasers = db.query(models.Teaser).all()
    return {"teasers": teasers}

@app.get("/teasers/{teaser_id}", response_model=schemas.TeaserResponse)
async def get_teaser(teaser_id: int, db: Session = Depends(get_db)):
    """
    Get a specific teaser by ID.
    """
    teaser = db.query(models.Teaser).filter(models.Teaser.id == teaser_id).first()
    if teaser is None:
        raise HTTPException(status_code=404, detail="Teaser not found")
    return teaser

@app.get("/teasers/{teaser_id}/report")
async def get_teaser_report(teaser_id: int, db: Session = Depends(get_db)):
    """
    Download the generated report for a teaser.
    """
    teaser = db.query(models.Teaser).filter(models.Teaser.id == teaser_id).first()
    if teaser is None:
        raise HTTPException(status_code=404, detail="Teaser not found")
    
    if teaser.status != TeaserStatus.COMPLETED or not teaser.report_path:
        raise HTTPException(status_code=404, detail="Report not yet available")
    
    # Check if the report file exists
    if not os.path.exists(teaser.report_path):
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=teaser.report_path,
        filename=f"{teaser.filename.replace('.pdf', '')}_report.pdf",
        media_type="application/pdf"
    )

@app.delete("/teasers/{teaser_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_teaser(teaser_id: int, db: Session = Depends(get_db)):
    """
    Delete a specific teaser by ID.
    """
    teaser = db.query(models.Teaser).filter(models.Teaser.id == teaser_id).first()
    if teaser is None:
        raise HTTPException(status_code=404, detail="Teaser not found")
    
    # If there's a report file, delete it
    if teaser.report_path and os.path.exists(teaser.report_path):
        os.remove(teaser.report_path)
    
    # Delete the teaser from the database
    db.delete(teaser)
    db.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/teasers/{teaser_id}/process", response_model=schemas.TeaserResponse)
async def process_teaser(
    teaser_id: int,
    process_request: schemas.TeaserProcessRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start processing a teaser with selected building blocks.
    """
    teaser = db.query(models.Teaser).filter(models.Teaser.id == teaser_id).first()
    if teaser is None:
        raise HTTPException(status_code=404, detail="Teaser not found")
    
    # Check if teaser is already being processed - return success instead of error
    if teaser.status == models.TeaserStatus.PROCESSING:
        return teaser
    
    # Update teaser status to PROCESSING
    teaser.status = models.TeaserStatus.PROCESSING
    db.commit()
    db.refresh(teaser)
    
    # Start the pipeline processing with selected building blocks
    pipeline = TeaserProcessingPipeline(db, nlp_processor)
    background_tasks.add_task(pipeline.process, teaser_id, process_request.building_blocks)
    
    return teaser

@app.post("/teasers/{teaser_id}/cancel", response_model=schemas.TeaserResponse)
async def cancel_teaser_processing(
    teaser_id: int,
    db: Session = Depends(get_db)
):
    """
    Cancel processing of a teaser.
    """
    teaser = db.query(models.Teaser).filter(models.Teaser.id == teaser_id).first()
    if teaser is None:
        raise HTTPException(status_code=404, detail="Teaser not found")
    
    # Can only cancel if teaser is currently processing
    if teaser.status != models.TeaserStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Only processing teasers can be canceled")
    
    # Update teaser status to ERROR
    teaser.status = models.TeaserStatus.ERROR
    db.commit()
    db.refresh(teaser)
    
    print(f"Processing canceled for teaser {teaser_id}")
    
    return teaser

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)