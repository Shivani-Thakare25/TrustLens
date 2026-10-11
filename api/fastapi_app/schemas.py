"""Pydantic request/response schemas for the FastAPI app."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ReviewRequest(BaseModel):
    review_text: str = Field(
        ..., max_length=20000, examples=["Best product ever!!! Highly recommend!"]
    )
    rating: Optional[float] = Field(None, ge=1, le=5)
    verified_purchase: Optional[bool] = None
    review_title: Optional[str] = None
    review_date: Optional[str] = None
    product_id: Optional[str] = None
    user_id: Optional[str] = None
    review_id: Optional[str] = None
    product_category: Optional[str] = None
    helpful_votes: Optional[int] = Field(None, ge=0)
    total_votes: Optional[int] = Field(None, ge=0)
    product_price: Optional[float] = Field(None, ge=0)
    image_base64: Optional[str] = Field(
        None, max_length=7_000_000, description="Optional JPEG/PNG/WebP (<=5 MB), base64-encoded"
    )

    @field_validator("review_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("review_text must not be blank")
        return v


class BatchRequest(BaseModel):
    reviews: list[ReviewRequest] = Field(..., min_length=1, max_length=500)


class PredictResponse(BaseModel):
    review_id: str
    classification: str
    fake_probability: float
    genuine_probability: float
    risk_score: int
    trust_score: int
    confidence: float
    decision_band: str
    overall_sentiment: str
    aspects: list[dict[str, Any]]
    similar_reviews: list[dict[str, Any]]
    evidence: dict[str, Any]
    explanation: dict[str, Any]
    counterfactuals: dict[str, Any]
    model_version: Optional[str]
    timestamp: str
    image_analysis: Optional[dict[str, Any]] = None
