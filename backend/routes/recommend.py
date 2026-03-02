from __future__ import annotations
"""Use-case recommendation engine."""
import re
from fastapi import APIRouter
from models import UseCaseRequest, ModelScore, RecommendationResponse
from data import get_all_models

router = APIRouter(prefix="/api", tags=["recommend"])

# Keyword-to-dimension mapping for scoring use cases
KEYWORD_MAP = {
    "code": ["code", "coding", "programming", "software", "developer", "debug", "refactor",
             "api", "function", "algorithm", "script", "python", "javascript", "java", "rust"],
    "reasoning": ["reason", "reasoning", "logic", "analyze", "analysis", "complex", "think",
                  "problem-solving", "decision", "strategy", "planning", "evaluate"],
    "math": ["math", "mathematics", "calculate", "equation", "statistics", "numerical",
             "algebra", "calculus", "proof", "theorem", "formula"],
    "creative": ["write", "writing", "creative", "story", "content", "blog", "marketing",
                 "copywriting", "email", "draft", "narrative", "poetry"],
    "long_context": ["long", "document", "book", "large", "pdf", "whitepaper", "research",
                     "paper", "summarize", "summary", "corpus", "dataset"],
    "multilingual": ["translate", "translation", "multilingual", "language", "spanish",
                     "french", "german", "chinese", "japanese", "localize"],
    "cost": ["cheap", "budget", "affordable", "cost", "economical", "free", "low-cost",
             "inexpensive", "save money", "pricing"],
    "speed": ["fast", "speed", "quick", "real-time", "latency", "responsive", "efficient",
              "high-throughput", "batch", "streaming"],
}

# Benchmark weights per dimension
DIMENSION_BENCHMARKS = {
    "quality": ["LMSYS Arena", "MMLU-Pro"],
    "code": ["LiveCodeBench"],
    "reasoning": ["GPQA Diamond", "MMLU-Pro"],
    "math": ["GPQA Diamond"], # GPQA contains hard math/science
    "creative": ["LMSYS Arena"],
}


def _detect_priorities(use_case: str) -> dict[str, float]:
    """Auto-detect dimensional weights from use-case text."""
    use_case_lower = use_case.lower()
    detected = {}
    for dimension, keywords in KEYWORD_MAP.items():
        matches = sum(1 for kw in keywords if kw in use_case_lower)
        if matches > 0:
            detected[dimension] = min(matches * 1.5, 5.0)
    return detected


def _score_model_dimension(model, dimension: str) -> float:
    """Score a model on a specific dimension (0-100)."""
    if dimension == "cost":
        # Invert cost — cheaper = higher score
        max_input = 200.0  # handle expensive models like o1-pro
        score = max(0, (1 - model.input_price_per_1m / max_input)) * 100
        return score

    if dimension == "speed":
        # Cheaper models tend to be faster; use output price as proxy
        max_output = 800.0  # handle expensive reasoning models
        score = max(0, (1 - model.output_price_per_1m / max_output)) * 100
        return score

    if dimension == "long_context":
        # Score based on context window size
        max_ctx = 2_000_000
        score = min((model.context_window / max_ctx) * 100, 100) if max_ctx > 0 else 0
        return min(score, 100)

    if dimension == "multilingual":
        # Mistral and Claude are strong here; use a base or curated score
        return 85.0 if "mistral" in model.provider.lower() or "anthropic" in model.provider.lower() else 75.0

    # For benchmark-based dimensions, average relevant benchmarks
    # Use 70.0 as default for frontier models to give them a baseline of intelligence
    benchmarks = DIMENSION_BENCHMARKS.get(dimension, ["MMLU-Pro"])
    scores = [model.benchmarks.get(bm, 70.0) for bm in benchmarks]
    return sum(scores) / len(scores) if scores else 70.0


def _generate_reasoning(model, dimension_scores: dict, priorities: dict) -> str:
    """Generate a human-readable reasoning string."""
    top_dims = sorted(priorities.items(), key=lambda x: x[1], reverse=True)[:3]
    parts = []

    for dim, weight in top_dims:
        if weight < 2:
            continue
        score = dimension_scores.get(dim, 50)
        label = dim.replace('_', ' ')
        if score >= 90:
            parts.append(f"State-of-the-art {label} performer ({score:.1f}/100)")
        elif score >= 80:
            parts.append(f"Top-tier {label} capability ({score:.1f}/100)")
        elif score >= 60:
            parts.append(f"Strong {label} capability ({score:.1f}/100)")
        else:
            parts.append(f"Competent {label} ({score:.1f}/100)")

    # Add historical context and benchmark citations
    if model.benchmarks:
        main_bm = "GPQA Diamond" if priorities.get("reasoning", 0) > 3 else "MMLU-Pro"
        if main_bm in model.benchmarks:
            parts.append(f"Cites a {model.benchmarks[main_bm]:.1f}% score on {main_bm}")

    if model.release_date:
        parts.append(f"Launched {model.release_date}")

    if model.strengths:
        parts.append(f"Strengths: {', '.join(model.strengths[:2])}")

    return ". ".join(parts) + "."


def _estimate_cost(model, use_case: str) -> str:
    """Generate a cost estimate string."""
    words = len(use_case.split())
    tokens_est = 100_000 # Default baseline
    label = "~100K tokens"

    if any(kw in use_case.lower() for kw in ["batch", "large", "dataset", "corpus", "document"]):
        tokens_est = 1_000_000
        label = "high-volume (~1M tokens)"
    elif any(kw in use_case.lower() for kw in ["api", "chat", "conversation", "real-time"]):
        tokens_est = 50_000
        label = "conversational (~50K tokens)"

    input_cost = tokens_est * model.input_price_per_token
    output_cost = (tokens_est * 0.3) * model.output_price_per_token
    total = input_cost + output_cost

    return (
        f"Estimated cost for {label}: ~${total:.4f}. "
        f"Precision rates: ${model.input_price_per_token:.8f}/input, "
        f"${model.output_price_per_token:.8f}/output token."
    )


@router.post("/recommend")
def recommend_model(request: UseCaseRequest) -> RecommendationResponse:
    """Score and rank models based on a use case and priorities."""
    models = get_all_models()

    # Merge user priorities with auto-detected priorities
    detected = _detect_priorities(request.use_case)
    priorities = {**request.priorities}
    for dim, score in detected.items():
        priorities[dim] = max(priorities.get(dim, 0), score)

    # Ensure all dimensions have a default
    all_dims = ["quality", "cost", "speed", "code", "reasoning", "math",
                "creative", "long_context", "multilingual"]
    for dim in all_dims:
        priorities.setdefault(dim, 2)

    scored_models: list[ModelScore] = []

    for model in models:
        # Apply hard filters
        if request.min_context_window and model.context_window < request.min_context_window:
            continue
        if request.budget_per_1m_tokens and model.input_price_per_1m > request.budget_per_1m_tokens:
            continue

        # Score each dimension
        dimension_scores = {}
        weighted_sum = 0
        total_weight = 0
        for dim in all_dims:
            score = _score_model_dimension(model, dim)
            dimension_scores[dim] = round(score, 1)
            weight = priorities.get(dim, 2)
            weighted_sum += score * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0
        reasoning = _generate_reasoning(model, dimension_scores, priorities)
        cost_est = _estimate_cost(model, request.use_case)

        scored_models.append(ModelScore(
            model=model,
            overall_score=round(overall, 1),
            dimension_scores=dimension_scores,
            reasoning=reasoning,
            cost_estimate=cost_est,
            match_percentage=round(overall, 1),
        ))

    # Sort by overall score descending
    scored_models.sort(key=lambda s: s.overall_score, reverse=True)

    if not scored_models:
        # Fallback if all filtered out
        scored_models = [ModelScore(
            model=models[0],
            overall_score=0,
            dimension_scores={},
            reasoning="No models matched your filters. Try relaxing your constraints.",
            cost_estimate="N/A",
            match_percentage=0,
        )]

    top = scored_models[0]
    alts = scored_models[1:5]

    summary = (
        f"Based on your use case, **{top.model.name}** by {top.model.provider} is the best match "
        f"with a {top.match_percentage:.0f}% compatibility score. "
        f"It offers {', '.join(top.model.strengths[:2])}. "
        f"{top.cost_estimate.split('.')[0]}."
    )

    return RecommendationResponse(
        top_pick=top,
        alternatives=alts,
        summary=summary,
    )
