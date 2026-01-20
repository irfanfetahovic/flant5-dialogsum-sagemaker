"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SummarizeRequest(BaseModel):
    """Request model for single summarization."""

    dialogue: str = Field(
        ...,
        description="The dialogue to summarize",
        min_length=10,
        max_length=4096,
        example="Customer: Hi, I need help with my order.\nAgent: Of course! What's the issue?",
    )
    max_tokens: Optional[int] = Field(
        default=200, description="Maximum tokens in summary", le=500, ge=50
    )

    class Config:
        schema_extra = {
            "example": {
                "dialogue": "Customer: Hi, I need help with my order.\nAgent: Of course! What's the issue?",
                "max_tokens": 200,
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for summarization."""

    dialogue: str = Field(description="Original dialogue")
    summary: str = Field(description="Generated summary")
    tokens_generated: int = Field(description="Number of tokens in summary")
    confidence: Optional[float] = Field(
        default=None, description="Confidence score (0-1)"
    )
    processing_time_ms: float = Field(description="Processing time in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "dialogue": "Customer: Hi, I need help with my order.\nAgent: Of course! What's the issue?",
                "summary": "Customer needs help with their order.",
                "tokens_generated": 7,
                "confidence": 0.85,
                "processing_time_ms": 1250.5,
            }
        }


class BatchSummarizeRequest(BaseModel):
    """Request model for batch summarization."""

    dialogues: List[str] = Field(
        ...,
        description="List of dialogues to summarize",
        min_items=1,
        max_items=100,
    )
    max_tokens: Optional[int] = Field(default=200, description="Max tokens per summary")

    class Config:
        schema_extra = {
            "example": {
                "dialogues": [
                    "Customer: Hi, I need help.\nAgent: Sure!",
                    "User: How does billing work?\nBot: We charge monthly.",
                ],
                "max_tokens": 200,
            }
        }


class BatchSummarizeResponse(BaseModel):
    """Response model for batch summarization."""

    results: List[SummarizeResponse] = Field(description="List of summaries")
    total_items: int = Field(description="Total items processed")
    successful: int = Field(description="Successfully processed items")
    failed: int = Field(description="Failed items")
    total_time_ms: float = Field(description="Total processing time")

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "dialogue": "...",
                        "summary": "...",
                        "tokens_generated": 7,
                        "confidence": 0.85,
                        "processing_time_ms": 1250.5,
                    }
                ],
                "total_items": 1,
                "successful": 1,
                "failed": 0,
                "total_time_ms": 1250.5,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status (healthy/degraded/unhealthy)")
    model_loaded: bool = Field(description="Is model loaded")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "timestamp": "2026-01-19T10:00:00",
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Additional details")
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracking"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid dialogue",
                "detail": "Dialogue must be at least 10 characters",
                "request_id": "req-123456",
            }
        }
