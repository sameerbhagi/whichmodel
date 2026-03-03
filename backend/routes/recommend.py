from __future__ import annotations
"""Use-case recommendation engine — matches models to specific use cases."""
import re
from fastapi import APIRouter
from models import UseCaseRequest, ModelScore, RecommendationResponse
from data import get_all_models

router = APIRouter(prefix="/api", tags=["recommend"])

# ── Keyword-to-dimension mapping ─────────────────────────────────────────────
KEYWORD_MAP = {
    "code": ["code", "coding", "programming", "software", "developer", "debug", "refactor",
             "api", "function", "algorithm", "script", "python", "javascript", "java", "rust",
             "build", "app", "application", "engineer", "github", "deploy", "ci/cd", "unit test",
             "testing", "backend", "frontend", "full-stack", "fullstack", "devops", "terminal",
             "ide", "autocomplete", "pull request", "pr review"],
    "reasoning": ["reason", "reasoning", "logic", "analyze", "analysis", "complex", "think",
                  "problem-solving", "decision", "strategy", "planning", "evaluate",
                  "critical thinking", "deduction", "inference", "hypothesis"],
    "math": ["math", "mathematics", "calculate", "equation", "statistics", "numerical",
             "algebra", "calculus", "proof", "theorem", "formula", "quantitative",
             "financial", "accounting", "modeling"],
    "creative": ["write", "writing", "creative", "story", "content", "blog", "marketing",
                 "copywriting", "email", "draft", "narrative", "poetry", "fiction",
                 "storytelling", "roleplay", "dialogue", "prose", "novel", "essay",
                 "brainstorm", "ad copy", "slogan", "social media post"],
    "long_context": ["long", "document", "book", "large", "pdf", "whitepaper", "research",
                     "paper", "summarize", "summary", "corpus", "dataset", "codebase",
                     "entire project", "monorepo", "legal", "contract", "regulation"],
    "multilingual": ["translate", "translation", "multilingual", "language", "spanish",
                     "french", "german", "chinese", "japanese", "localize", "localization",
                     "international", "global audience"],
    "cost": ["cheap", "budget", "affordable", "cost", "economical", "free", "low-cost",
             "inexpensive", "save money", "pricing", "high-volume", "bulk", "mass",
             "scale", "millions of requests"],
    "speed": ["fast", "speed", "quick", "real-time", "latency", "responsive", "efficient",
              "high-throughput", "batch", "streaming", "instant", "sub-second",
              "low-latency", "interactive"],
    "vision": ["image", "vision", "visual", "picture", "photo", "screenshot", "diagram",
               "chart", "ocr", "scan", "graphic", "ui", "design", "video"],
    "agent": ["agent", "agentic", "autonomous", "tool use", "function calling",
              "multi-step", "workflow", "orchestration", "automation", "pipeline"],
    "chat": ["chat", "chatbot", "conversation", "customer service", "support",
             "assistant", "help desk", "dialogue", "interactive"],
}

# ── Benchmark weights per dimension ──────────────────────────────────────────
DIMENSION_BENCHMARKS = {
    "quality":   ["LMSYS Arena", "MMLU-Pro"],
    "code":      ["LiveCodeBench"],
    "reasoning": ["GPQA Diamond", "MMLU-Pro"],
    "math":      ["GPQA Diamond"],
    "creative":  ["LMSYS Arena"],
    "agent":     ["LiveCodeBench", "LMSYS Arena"],
}

# ── Use-case category → model affinity ───────────────────────────────────────
# Maps detected use-case categories to OpenRouter model IDs that are specialist fits
MODEL_SPECIALTIES: dict[str, list[str]] = {
    "code": [
        "mistralai/devstral-2512", "mistralai/devstral-medium", "mistralai/devstral-small",
        "mistralai/codestral-2508",
        "openai/gpt-5.3-codex", "openai/gpt-5.2-codex", "openai/gpt-5.1-codex",
        "openai/gpt-5.1-codex-max", "openai/gpt-5.1-codex-mini",
        "openai/o3-mini", "openai/o4-mini",
        "anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.5",
        "anthropic/claude-3.7-sonnet",
    ],
    "creative": [
        "anthropic/claude-opus-4.6", "anthropic/claude-opus-4.5", "anthropic/claude-opus-4",
        "anthropic/claude-opus-4.1",
        "mistralai/mistral-small-creative",
        "anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.5",
    ],
    "reasoning": [
        "openai/o1", "openai/o3-mini", "openai/o4-mini",
        "google/gemini-3.1-pro-preview", "google/gemini-3-pro-preview",
        "anthropic/claude-opus-4.6", "anthropic/claude-opus-4.5",
        "anthropic/claude-3.7-sonnet:thinking",
    ],
    "math": [
        "openai/o1", "openai/o3-mini", "openai/o4-mini",
        "google/gemini-3.1-pro-preview", "google/gemini-2.5-pro",
        "anthropic/claude-3.7-sonnet:thinking",
    ],
    "long_context": [
        "google/gemini-2.5-pro", "google/gemini-2.5-flash",
        "google/gemini-3.1-pro-preview", "google/gemini-3-pro-preview",
        "anthropic/claude-sonnet-4.6",
    ],
    "cost": [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku", "anthropic/claude-3.5-haiku",
        "google/gemini-2.0-flash-001", "google/gemini-2.5-flash-lite",
        "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.2-3b-instruct",
        "mistralai/ministral-3b-2512", "mistralai/ministral-8b-2512",
        "google/gemma-3-4b-it",
    ],
    "speed": [
        "anthropic/claude-3-haiku", "anthropic/claude-3.5-haiku",
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.0-flash-001", "google/gemini-2.5-flash",
        "google/gemini-2.5-flash-lite",
        "openai/gpt-4o-mini",
    ],
    "multilingual": [
        "mistralai/mistral-large-2512", "mistralai/mistral-large",
        "mistralai/mistral-medium-3.1", "mistralai/mistral-medium-3",
        "anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.6",
        "google/gemma-3-27b-it",
    ],
    "vision": [
        "google/gemini-2.5-pro", "google/gemini-2.5-flash-image",
        "google/gemini-3-pro-image-preview", "google/gemini-3.1-flash-image-preview",
        "openai/gpt-4o", "openai/gpt-5.2",
        "meta-llama/llama-3.2-11b-vision-instruct",
    ],
    "agent": [
        "anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.6",
        "anthropic/claude-opus-4.5", "anthropic/claude-opus-4.6",
        "mistralai/devstral-2512", "mistralai/devstral-medium",
        "openai/gpt-5.3-codex", "openai/o4-mini",
    ],
    "chat": [
        "openai/gpt-4o", "openai/gpt-4o-mini",
        "openai/gpt-5.1-chat", "openai/gpt-5.2-chat",
        "anthropic/claude-haiku-4.5", "anthropic/claude-3.5-haiku",
        "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4.5",
        "meta-llama/llama-3.3-70b-instruct",
    ],
}


def _detect_priorities(use_case: str) -> dict[str, float]:
    """Auto-detect dimensional weights from use-case text.

    Uses stronger weight scaling so detected dimensions dominate over defaults.
    """
    use_case_lower = use_case.lower()
    detected = {}
    for dimension, keywords in KEYWORD_MAP.items():
        matches = sum(1 for kw in keywords if kw in use_case_lower)
        if matches > 0:
            # Strong scaling: 1 match = 3.0, 2+ matches = 4.0-5.0
            detected[dimension] = min(2.0 + matches * 1.5, 5.0)
    return detected


def _score_model_dimension(model, dimension: str) -> float:
    """Score a model on a specific dimension (0-100)."""
    if dimension == "cost":
        # Invert cost — cheaper = higher score (log-scale friendly)
        price = model.input_price_per_1m
        if price <= 0:
            return 95.0
        if price <= 0.5:
            return 95.0
        if price <= 2.0:
            return 85.0
        if price <= 5.0:
            return 70.0
        if price <= 15.0:
            return 50.0
        if price <= 50.0:
            return 30.0
        return 10.0

    if dimension == "speed":
        # Smaller/cheaper models tend to be faster; use a tiered system
        price = model.input_price_per_1m + model.output_price_per_1m
        if price <= 2.0:
            return 95.0
        if price <= 8.0:
            return 80.0
        if price <= 20.0:
            return 60.0
        if price <= 60.0:
            return 40.0
        return 20.0

    if dimension == "long_context":
        ctx = model.context_window
        if ctx >= 2_000_000:
            return 100.0
        if ctx >= 1_000_000:
            return 90.0
        if ctx >= 200_000:
            return 70.0
        if ctx >= 128_000:
            return 55.0
        if ctx >= 32_000:
            return 35.0
        return 15.0

    if dimension == "multilingual":
        provider = model.provider.lower()
        if "mistral" in provider:
            return 92.0
        if "anthropic" in provider:
            return 85.0
        if "google" in provider:
            return 82.0
        if "openai" in provider:
            return 78.0
        if "meta" in provider:
            return 75.0
        return 60.0

    if dimension == "vision":
        has_vision = "image" in model.input_modalities or "video" in model.input_modalities
        if has_vision:
            # Models with vision + high benchmarks score highest
            arena = model.benchmarks.get("LMSYS Arena", 0)
            return min(70 + (arena - 80) * 2, 100) if arena > 0 else 75.0
        return 20.0  # Text-only models score low on vision

    if dimension == "agent":
        # Agent capability correlates with coding + reasoning benchmarks
        code_score = model.benchmarks.get("LiveCodeBench", 0)
        arena_score = model.benchmarks.get("LMSYS Arena", 0)
        if code_score > 0 and arena_score > 0:
            return (code_score * 0.6 + arena_score * 0.4)
        if code_score > 0:
            return code_score
        return 35.0  # Unknown agent capability

    if dimension == "chat":
        arena = model.benchmarks.get("LMSYS Arena", 0)
        if arena > 0:
            return arena
        return 50.0

    # For benchmark-based dimensions, use real data or a low default
    benchmarks = DIMENSION_BENCHMARKS.get(dimension, ["MMLU-Pro"])
    scores = [model.benchmarks.get(bm) for bm in benchmarks]
    real_scores = [s for s in scores if s is not None]

    if real_scores:
        return sum(real_scores) / len(real_scores)
    # No benchmark data = unknown quality → low score
    return 40.0


def _use_case_relevance(model, use_case_lower: str, detected_dims: dict) -> float:
    """Score 0-20 bonus based on how well model's curated metadata matches the use case.

    This is the key differentiator: models with curated use_cases and strengths
    that overlap with the user's query get a significant bonus.
    """
    bonus = 0.0

    # 1. Check model's curated use_cases against the user's text (up to +10)
    for uc in model.use_cases:
        uc_words = set(uc.lower().split())
        query_words = set(use_case_lower.split())
        overlap = uc_words & query_words
        # Meaningful overlap (not just "and", "for", etc.)
        meaningful = {w for w in overlap if len(w) > 3}
        if meaningful:
            bonus += min(len(meaningful) * 2.0, 4.0)

    # 2. Check model's strengths against the use case (up to +6)
    for strength in model.strengths:
        strength_lower = strength.lower()
        # Check if any significant words from the strength appear in the use case
        strength_words = [w for w in strength_lower.split() if len(w) > 3]
        for word in strength_words:
            if word in use_case_lower:
                bonus += 1.5
                break

    # 3. Specialist affinity — model is in the specialty list for detected dims (up to +8)
    for dim, weight in detected_dims.items():
        if weight < 3.0:
            continue
        specialists = MODEL_SPECIALTIES.get(dim, [])
        if model.openrouter_id in specialists:
            # Weight by how strongly this dimension was detected
            bonus += min(weight * 1.5, 8.0)
            break  # Only count the strongest match

    return min(bonus, 20.0)


def _generate_reasoning(model, dimension_scores: dict, priorities: dict,
                        relevance_bonus: float) -> str:
    """Generate a human-readable reasoning string."""
    # Focus on the dimensions the user cares about most
    top_dims = sorted(priorities.items(), key=lambda x: x[1], reverse=True)[:3]
    parts = []

    for dim, weight in top_dims:
        if weight < 2.5:
            continue
        score = dimension_scores.get(dim, 50)
        label = dim.replace('_', ' ')
        if score >= 90:
            parts.append(f"Exceptional {label} ({score:.0f}/100)")
        elif score >= 80:
            parts.append(f"Top-tier {label} ({score:.0f}/100)")
        elif score >= 65:
            parts.append(f"Strong {label} ({score:.0f}/100)")
        elif score >= 50:
            parts.append(f"Solid {label} ({score:.0f}/100)")
        else:
            parts.append(f"Moderate {label} ({score:.0f}/100)")

    # Specialist match callout
    if relevance_bonus >= 8:
        parts.insert(0, "Specialist match for your use case")
    elif relevance_bonus >= 4:
        parts.insert(0, "Strong alignment with your requirements")

    # Benchmark citation
    if model.benchmarks:
        # Pick the most relevant benchmark
        main_bm = None
        if priorities.get("code", 0) >= 3.5:
            main_bm = "LiveCodeBench"
        elif priorities.get("reasoning", 0) >= 3.5 or priorities.get("math", 0) >= 3.5:
            main_bm = "GPQA Diamond"
        else:
            main_bm = "MMLU-Pro"
        if main_bm and main_bm in model.benchmarks:
            parts.append(f"{model.benchmarks[main_bm]:.1f}% on {main_bm}")

    if model.strengths:
        parts.append(f"Key strengths: {', '.join(model.strengths[:2])}")

    if model.release_date:
        parts.append(f"Released {model.release_date}")

    return ". ".join(parts) + "."


def _estimate_cost(model, use_case: str) -> str:
    """Generate a cost estimate string."""
    tokens_est = 100_000
    label = "~100K tokens"

    if any(kw in use_case.lower() for kw in ["batch", "large", "dataset", "corpus", "document",
                                               "codebase", "project", "monorepo"]):
        tokens_est = 1_000_000
        label = "high-volume (~1M tokens)"
    elif any(kw in use_case.lower() for kw in ["api", "chat", "conversation", "real-time",
                                                 "interactive", "assistant"]):
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

    # Ensure all dimensions have a low default (so they don't overwhelm detected ones)
    all_dims = ["quality", "cost", "speed", "code", "reasoning", "math",
                "creative", "long_context", "multilingual", "vision", "agent", "chat"]
    for dim in all_dims:
        priorities.setdefault(dim, 1.0)

    use_case_lower = request.use_case.lower()
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
            weight = priorities.get(dim, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        base_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Use-case relevance bonus — the key differentiator
        relevance = _use_case_relevance(model, use_case_lower, detected)
        overall = min(base_score + relevance, 100.0)

        reasoning = _generate_reasoning(model, dimension_scores, priorities, relevance)
        cost_est = _estimate_cost(model, request.use_case)

        # Only show most-relevant dimension scores to the user
        top_priority_dims = sorted(priorities.items(), key=lambda x: x[1], reverse=True)[:6]
        filtered_scores = {dim: dimension_scores[dim] for dim, _ in top_priority_dims
                          if dim in dimension_scores}

        scored_models.append(ModelScore(
            model=model,
            overall_score=round(overall, 1),
            dimension_scores=filtered_scores,
            reasoning=reasoning,
            cost_estimate=cost_est,
            match_percentage=round(overall, 1),
        ))

    # Sort by overall score descending
    scored_models.sort(key=lambda s: s.overall_score, reverse=True)

    if not scored_models:
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
        f"It offers {', '.join(top.model.strengths[:2]) if top.model.strengths else 'strong overall performance'}. "
        f"{top.cost_estimate.split('.')[0]}."
    )

    return RecommendationResponse(
        top_pick=top,
        alternatives=alts,
        summary=summary,
    )
