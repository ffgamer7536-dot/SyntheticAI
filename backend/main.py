import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from routers.inference import router as inference_router
from services.inference_service import get_inference_service


# Initialize FastAPI app
app = FastAPI(
    title="Offroad Segmentation Inference API",
    description="Real-time semantic segmentation for offroad terrain classification",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference_router, prefix="/api")


# Startup event - initialize model
@app.on_event("startup")
async def startup_event():
    """Initialize inference service on startup."""
    try:
        print("Initializing inference service...")
        get_inference_service()
        print("Inference service initialized successfully")
    except Exception as e:
        print(f"Failed to initialize inference service: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Offroad Segmentation Inference API"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Offroad Segmentation Inference API",
        "docs": "/docs",
        "openapi_schema": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
