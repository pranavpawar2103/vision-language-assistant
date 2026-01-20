# src/api/main.py
"""
FastAPI Backend for Vision-Language Assistant
Author: Pranav
Date: January 2025

RESTful API providing image analysis endpoints
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import shutil
import os
from datetime import datetime

from src.pipeline import VisionLanguagePipeline

# Initialize FastAPI app
app = FastAPI(
    title="Vision-Language Assistant API",
    description="AI-powered product image analysis combining computer vision and LLMs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (global instance for efficiency)
# This loads models once when server starts, not per-request
print("🚀 Initializing Vision-Language Pipeline...")
pipeline = VisionLanguagePipeline(use_api=True)
print("✅ API Server Ready!\n")

# Create uploads directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Pydantic models for request/response validation
class AnalysisResponse(BaseModel):
    """Response model for image analysis"""
    success: bool
    image_filename: str
    vision_analysis: dict
    llm_outputs: Optional[dict] = None
    processing_time: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    api_available: bool


# API Endpoints

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vision-Language Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze product image",
            "POST /analyze/batch": "Batch analysis of multiple images",
            "GET /health": "Health check",
            "GET /docs": "API documentation (Swagger UI)"
        },
        "author": "Pranav",
        "github": "https://github.com/yourusername/vision-language-assistant"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Used by:
    - Deployment platforms (Render, Railway, etc.)
    - Monitoring systems
    - Load balancers
    """
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        api_available=pipeline.llm is not None
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Product image to analyze"),
    tasks: Optional[str] = Form(
        default="description,alt_text,structured_data",
        description="Comma-separated list of tasks: description, alt_text, structured_data"
    )
):
    """
    Analyze a single product image
    
    **Tasks:**
    - `description`: Generate product description
    - `alt_text`: Generate accessibility alt text
    - `structured_data`: Extract structured JSON data
    
    **Example:**
```bash
    curl -X POST "http://localhost:8000/analyze" \\
         -F "file=@shoe.jpg" \\
         -F "tasks=description,alt_text"
```
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = Path(file.filename).suffix
        save_filename = f"{timestamp}_{file.filename}"
        save_path = UPLOAD_DIR / save_filename
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse tasks
        task_list = [t.strip() for t in tasks.split(",")]
        
        # Run analysis
        result = pipeline.analyze_product(
            str(save_path),
            tasks=task_list
        )
        
        # Build response
        response = AnalysisResponse(
            success=True,
            image_filename=file.filename,
            vision_analysis=result["vision_analysis"],
            llm_outputs=result.get("llm_outputs"),
            processing_time=result["processing_time_seconds"],
            timestamp=datetime.now().isoformat()
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/batch")
async def batch_analyze(
    files: List[UploadFile] = File(..., description="Multiple product images")
):
    """
    Batch analyze multiple images
    
    **Example:**
```bash
    curl -X POST "http://localhost:8000/analyze/batch" \\
         -F "files=@shoe1.jpg" \\
         -F "files=@shoe2.jpg" \\
         -F "files=@shoe3.jpg"
```
    """
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request"
        )
    
    results = []
    saved_paths = []
    
    try:
        # Save all files
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{timestamp}_{file.filename}"
            save_path = UPLOAD_DIR / save_filename
            
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_paths.append(str(save_path))
        
        # Batch process
        batch_results = pipeline.batch_analyze(
            saved_paths,
            save_results=False
        )
        
        # Format responses
        for i, result in enumerate(batch_results):
            results.append({
                "image_filename": files[i].filename,
                "vision_analysis": result["vision_analysis"],
                "llm_outputs": result.get("llm_outputs"),
                "processing_time": result["processing_time_seconds"]
            })
        
        return JSONResponse({
            "success": True,
            "total_images": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@app.post("/qa")
async def visual_question_answering(
    file: UploadFile = File(...),
    question: str = Form(..., description="Question about the image")
):
    """
    Answer a question about an image
    
    **Example:**
```bash
    curl -X POST "http://localhost:8000/qa" \\
         -F "file=@shoe.jpg" \\
         -F "question=What color is this product?"
```
    """
    
    try:
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{timestamp}_{file.filename}"
        save_path = UPLOAD_DIR / save_filename
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get vision analysis
        vision_result = pipeline.vision.get_product_info(str(save_path))
        
        # Answer question
        if pipeline.llm:
            answer = pipeline.llm.answer_visual_question(
                vision_result['description'],
                question,
                vision_result['detected_objects']
            )
        else:
            answer = f"Based on the image showing {vision_result['description']}, the answer depends on specific details."
        
        return JSONResponse({
            "success": True,
            "question": question,
            "answer": answer,
            "vision_analysis": vision_result
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"QA failed: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """
    Get API usage statistics
    """
    if pipeline.llm:
        stats = pipeline.llm.get_usage_stats()
        return JSONResponse({
            "success": True,
            "stats": stats,
            "note": "Statistics are reset when server restarts"
        })
    else:
        return JSONResponse({
            "success": False,
            "message": "LLM not configured"
        })


# Run server
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STARTING FASTAPI SERVER")
    print("="*70)
    print("\n📍 Server will be available at:")
    print("   - Local: http://localhost:8000")
    print("   - Docs:  http://localhost:8000/docs")
    print("\n💡 Press CTRL+C to stop\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )