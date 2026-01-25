"""
FastAPI application for dialog summarization inference.
Production-ready with error handling, logging, and monitoring.

Environment Variables (can be set via export command, .env file, or platform config)
  - MODEL_ID: HuggingFace model to load (default: google/flan-t5-base)
  - PEFT_WEIGHTS_PATH: Path to LoRA weights (local or S3). If provided, LoRA adapter will be loaded and merged.
                       Example: s3://bucket/path/to/adapter or /local/path/to/adapter
  - API_KEY: Optional API key for request authentication
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZIPMiddleware
import torch

from api.models import (
    SummarizeRequest,
    SummarizeResponse,
    BatchSummarizeRequest,
    BatchSummarizeResponse,
    HealthResponse,
    HealthDetailedResponse,
    ErrorResponse,
)
from src.inference import load_base_model, summarize_dialogue

# Configure logging
# Send INFO and above to console (only done once)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Set up global logger with module name so it is known where logs originate
logger = logging.getLogger(__name__)

# Global model state (to load model once on startup)
_model = None
_tokenizer = None
_model_loaded = False
_model_config = {}


def load_model_with_peft():
    """
    Load base model and optionally merge LoRA weights.

    Supports:
    - Base model only (MODEL_ID)
    - Base model + LoRA from local filesystem
    - Base model + LoRA from S3 bucket

    Returns:
        tuple: (model, tokenizer)
    """
    global _model_config

    model_id = os.getenv("MODEL_ID", "google/flan-t5-base")
    peft_weights_path = os.getenv("PEFT_WEIGHTS_PATH")
    logger.info(f"Loading base model: {model_id}")
    model, tokenizer = load_base_model(model_id)

    _model_config = {
        "model_id": model_id,
        "lora_merged": False,  # True if LoRA weights were merged into base model
        "lora_source": None,  # Source path/URI of LoRA weights (metadata only after merge)
    }

    if peft_weights_path:
        logger.info(f"Loading LoRA weights from: {peft_weights_path}")
        try:
            from peft import PeftModel

            # Handle S3 paths
            if peft_weights_path.startswith("s3://"):
                logger.info("Downloading LoRA weights from S3...")
                import tempfile
                import boto3

                s3_client = boto3.client("s3")

                # Parse S3 path (because boto3 does not accept s3:// URIs directly)
                bucket, key = peft_weights_path.replace("s3://", "").split("/", 1)
                # Create temporary directory to download files. It is needed because PeftModel expects a local path.
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_path = os.path.join(tmpdir, "adapter")
                    os.makedirs(local_path, exist_ok=True)

                    # Download all files in the prefix
                    # Pagination is a process of iterating through multiple pages of results from S3
                    # It is used because S3 can return a limited number of objects per request (usually 1000)
                    # Example: One page contains 1000 objects, next page contains the next 1000, etc.

                    # I want to list files from S3, but safely, page by page.
                    paginator = s3_client.get_paginator("list_objects_v2")
                    # Iterate through each page of results. Each page is a dictionary with metadata and 'Contents' key.
                    # Example: {'Contents': [{'Key': 'path/to/file1'}, {'Key': 'path/to/file2'}], ...}
                    for page in paginator.paginate(Bucket=bucket, Prefix=key):
                        for obj in page.get("Contents", []):
                            s3_key = obj["Key"]
                            # Return only the filename to save in local_path (after the last '/')
                            filename = os.path.basename(s3_key)
                            file_path = os.path.join(local_path, filename)
                            logger.info(f"Downloading: {s3_key} -> {file_path}")
                            s3_client.download_file(bucket, s3_key, file_path)

                    # Load PEFT model from local copy
                    model = PeftModel.from_pretrained(model, local_path)
                    logger.info("LoRA weights loaded and merged from S3")
            else:
                # Load from local filesystem
                logger.info(
                    f"Loading LoRA weights from local path: {peft_weights_path}"
                )
                model = PeftModel.from_pretrained(model, peft_weights_path)
                logger.info("LoRA weights loaded and merged from filesystem")

            # Merge LoRA weights into base model for inference
            model = model.merge_and_unload()
            logger.info("LoRA weights merged into base model")

            _model_config["lora_merged"] = True
            _model_config["lora_source"] = peft_weights_path

        except ImportError:
            logger.warning("PEFT library not available, loading base model only")
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            logger.warning("Continuing with base model only")

    return model, tokenizer


# Decorator function to extend the functionality of FastAPI app lifespan function
@asynccontextmanager
# Function to manage the lifespan of the FastAPI application.
# It is called on startup by FastAPI to initialize resources and on shutdown to clean up.
async def lifespan(app: FastAPI):
    """Manage app lifecycle - load model on startup, cleanup on shutdown."""
    global _model, _tokenizer, _model_loaded

    # Startup
    logger.info("=" * 60)
    logger.info("STARTUP: Dialog Summarization API")
    logger.info("=" * 60)
    try:
        _model, _tokenizer = load_model_with_peft()
        _model_loaded = True
        logger.info("Model loaded successfully")
        logger.info(f"   Model Config: {_model_config}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        _model_loaded = False
    # yield control back to FastAPI to start serving requests
    yield

    # Shutdown part - cleanup resources. It is called when the app is shutting down.
    logger.info("=" * 60)
    logger.info("SHUTDOWN: Cleaning up resources...")
    logger.info("=" * 60)
    try:
        if _model is not None:
            del _model
        if _tokenizer is not None:
            del _tokenizer
        torch.cuda.empty_cache()
        logger.info("Cleanup complete")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


# Create FastAPI app
app = FastAPI(
    title="Dialog Summarization API",
    description="AI-powered conversation summarization using FLAN-T5 and LoRA",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware (code that runs before/after each request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Change to specific domains in production, right now it allows all
    allow_credentials=True,  # Allow cookies, authorization headers, etc.
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Add compression to HTTP responses, so they take less bandwidth when sent to clients
app.add_middleware(GZIPMiddleware, minimum_size=1000)


# Custom exception handler - called by FastAPI when unhandled exceptions occur in endpoints (routes)
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.error(f"[{request_id}] Unexpected error: {exc}", exc_info=True)

    error = ErrorResponse(
        error="Internal server error",
        detail=str(exc) if logger.level == logging.DEBUG else None,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=500,
        # Convert Pydantic model to dict for JSON response
        content=error.model_dump(),
    )


# Decorator that tells FastAPI to use the function bellow as middleware for all HTTP requests
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracking."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Routes
@app.get("/", tags=["Info"])
async def root():
    """API root endpoint with documentation links."""
    return {
        "name": "Dialog Summarization API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "summarize": "POST /summarize - Single summarization",
            "batch": "POST /batch - Batch summarization",
            "health": "GET /health - Health check",
        },
    }


@app.get("/health", tags=["Health"], response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    from datetime import datetime

    if not _model_loaded:
        logger.warning("Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded - service unavailable",
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0",
        timestamp=datetime.now(),
    )


@app.get("/health/detailed", tags=["Health"], response_model=HealthDetailedResponse)
async def health_check_detailed():
    """Detailed health check with model configuration."""
    from datetime import datetime

    if not _model_loaded:
        return HealthDetailedResponse(
            status="unhealthy",
            model_loaded=False,
            model_config={},
            cuda_available=torch.cuda.is_available(),
            version="1.0.0",
            timestamp=datetime.now(),
        )

    return HealthDetailedResponse(
        status="healthy",
        model_loaded=True,
        model_config=_model_config,
        cuda_available=torch.cuda.is_available(),
        version="1.0.0",
        timestamp=datetime.now(),
    )


@app.post("/summarize", tags=["Summarization"], response_model=SummarizeResponse)
async def summarize_single(request: SummarizeRequest):
    """
    Summarize a single dialogue.

    **Parameters:**
    - `dialogue`: The conversation text to summarize (10-4096 chars)
    - `max_tokens`: Maximum tokens in output (default: 200, max: 500)

    **Returns:**
    - `summary`: Generated summary
    - `tokens_generated`: Number of tokens in summary
    - `processing_time_ms`: Time taken to generate
    """
    request_id = uuid.uuid4()
    logger.info(f"[{request_id}] Summarization request: {len(request.dialogue)} chars")

    # Validate model
    if not _model_loaded or _model is None:
        logger.error(f"[{request_id}] Model not loaded")
        raise HTTPException(
            status_code=503, detail="Model not available - service initializing"
        )

    try:
        start_time = time.time()

        # Generate summary
        summary = summarize_dialogue(
            _model, _tokenizer, request.dialogue, max_new_tokens=request.max_tokens
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"[{request_id}] Summary generated in {processing_time:.1f}ms")

        return SummarizeResponse(
            dialogue=request.dialogue,
            summary=summary,
            tokens_generated=len(summary.split()),
            confidence=0.85,  # Placeholder - could be enhanced with real confidence scoring
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error during summarization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}",
        )


@app.post("/batch", tags=["Summarization"], response_model=BatchSummarizeResponse)
async def summarize_batch(request: BatchSummarizeRequest):
    """
    Summarize multiple dialogues in batch.

    **Parameters:**
    - `dialogues`: List of conversations (1-100 items)
    - `max_tokens`: Maximum tokens per summary (default: 200)

    **Returns:**
    - `results`: List of individual summaries
    - `successful`: Number of successful summaries
    - `failed`: Number of failed summaries
    - `total_time_ms`: Total processing time
    """
    request_id = uuid.uuid4()
    logger.info(f"[{request_id}] Batch request: {len(request.dialogues)} dialogues")

    if not _model_loaded or _model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    for idx, dialogue in enumerate(request.dialogues):
        try:
            item_start = time.time()

            # Summarize
            summary = summarize_dialogue(
                _model,
                _tokenizer,
                dialogue,
                max_new_tokens=request.max_tokens,
            )

            item_time = (time.time() - item_start) * 1000

            results.append(
                SummarizeResponse(
                    dialogue=dialogue,
                    summary=summary,
                    tokens_generated=len(summary.split()),
                    confidence=0.85,
                    processing_time_ms=item_time,
                )
            )
            successful += 1

        except Exception as e:
            logger.error(f"[{request_id}] Item {idx}: Error - {e}")
            failed += 1
            continue

    total_time = (time.time() - start_time) * 1000

    logger.info(
        f"[{request_id}] Batch complete: {successful} successful, {failed} failed in {total_time:.1f}ms"
    )

    return BatchSummarizeResponse(
        results=results,
        total_items=len(request.dialogues),
        successful=successful,
        failed=failed,
        total_time_ms=total_time,
    )


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get model and system statistics."""
    return {
        "model": {
            "loaded": _model_loaded,
            "config": _model_config if _model_loaded else {},
        },
        "system": {
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
