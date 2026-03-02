from __future__ import annotations
"""Pydantic models for WhichModel API."""
from pydantic import BaseModel, Field
from typing import Optional


class Benchmark(BaseModel):
    """A single benchmark score."""
    name: str
    score: float
    max_score: float = 100.0
    category: str = "general"  # general, code, math, reasoning, multimodal


class LLMModel(BaseModel):
    """Represents a frontier LLM model."""
    id: str
    openrouter_id: str = ""  # original OpenRouter ID (e.g. "openai/gpt-4o")
    name: str
    provider: str
    description: str
    parameters_b: Optional[float] = None  # billions of parameters
    context_window: int  # tokens
    input_price_per_1m: float  # USD per 1M input tokens
    output_price_per_1m: float  # USD per 1M output tokens
    max_output_tokens: int = 4096
    release_date: str = ""
    update_history: list[str] = []
    benchmarks: dict[str, float] = {}  # benchmark_name -> score (0-100 normalized)
    strengths: list[str] = []
    weaknesses: list[str] = []
    use_cases: list[str] = []
    logo_color: str = "#6366f1"  # for UI display
    input_modalities: list[str] = ["text"]
    output_modalities: list[str] = ["text"]
    modality: str = "text->text"
    # Pricing per token (for UI display)
    input_price_per_token: float = 0.0
    output_price_per_token: float = 0.0


class UseCaseRequest(BaseModel):
    """Request body for use-case recommendation."""
    use_case: str = Field(..., min_length=10, description="Description of your use case")
    priorities: dict[str, int] = Field(
        default_factory=lambda: {
            "quality": 3,
            "cost": 3,
            "speed": 3,
            "code": 3,
            "reasoning": 3,
            "long_context": 3,
        },
        description="Priority weights from 1-5 for each dimension"
    )
    budget_per_1m_tokens: Optional[float] = None
    min_context_window: Optional[int] = None


class ModelScore(BaseModel):
    """A scored model in recommendation results."""
    model: LLMModel
    overall_score: float
    dimension_scores: dict[str, float]
    reasoning: str
    cost_estimate: str
    match_percentage: float


class RecommendationResponse(BaseModel):
    """Response body for use-case recommendation."""
    top_pick: ModelScore
    alternatives: list[ModelScore]
    summary: str
