from __future__ import annotations
"""Use-case recommendation engine — dynamically matches models to specific use cases
using each model's curated metadata (use_cases, strengths, description, benchmarks)."""
import re
from fastapi import APIRouter
from models import UseCaseRequest, ModelScore, RecommendationResponse
from data import get_all_models

router = APIRouter(prefix="/api", tags=["recommend"])

# ── Common stop words to ignore during text matching ─────────────────────────
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "it", "its", "that", "this", "are", "was",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall", "not", "no",
    "so", "if", "as", "up", "out", "than", "then", "into", "over", "such",
    "very", "just", "also", "more", "most", "some", "any", "all", "each",
    "use", "used", "using", "need", "needs", "want", "like", "i", "my", "me",
})

# ── Keyword-to-dimension mapping ─────────────────────────────────────────────
KEYWORD_MAP = {
    "code": ["code", "coding", "programming", "software", "developer", "debug", "refactor",
             "api", "function", "algorithm", "script", "python", "javascript", "java", "rust",
             "build", "app", "application", "engineer", "github", "deploy", "ci/cd", "unit test",
             "testing", "backend", "frontend", "full-stack", "fullstack", "devops", "terminal",
             "ide", "autocomplete", "pull request", "pr review", "codebase", "repo"],
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


def _tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase words from text, removing stop words."""
    words = set(re.findall(r'[a-z][a-z0-9\-]+', text.lower()))
    return words - STOP_WORDS


def _phrase_match_score(query_lower: str, target_text: str) -> float:
    """Score how well a target text phrase matches the query.

    Uses both:
    - Full phrase substring matching (highest value)
    - Individual meaningful word overlap
    """
    target_lower = target_text.lower()
    score = 0.0

    # Full phrase match — e.g. "coding assistant" found in model's use_cases
    if target_lower in query_lower:
        score += 8.0
    elif query_lower in target_lower:
        score += 6.0

    # Word-level overlap
    query_words = _tokenize(query_lower)
    target_words = _tokenize(target_lower)
    if not target_words:
        return score

    overlap = query_words & target_words
    if overlap:
        # Score by fraction of target words matched, weighted by count
        match_ratio = len(overlap) / len(target_words)
        score += match_ratio * 4.0 + len(overlap) * 1.0

    return score


def _detect_priorities(use_case: str) -> dict[str, float]:
    """Auto-detect dimensional weights from use-case text.

    Uses stronger weight scaling so detected dimensions dominate over defaults.
    """
    use_case_lower = use_case.lower()
    detected = {}
    for dimension, keywords in KEYWORD_MAP.items():
        matches = sum(1 for kw in keywords if kw in use_case_lower)
        if matches > 0:
            detected[dimension] = min(2.0 + matches * 1.5, 5.0)
    return detected


def _score_model_dimension(model, dimension: str) -> float:
    """Score a model on a specific dimension (0-100)."""
    if dimension == "cost":
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
            arena = model.benchmarks.get("LMSYS Arena", 0)
            return min(70 + (arena - 80) * 2, 100) if arena > 0 else 75.0
        return 20.0

    if dimension == "agent":
        code_score = model.benchmarks.get("LiveCodeBench", 0)
        arena_score = model.benchmarks.get("LMSYS Arena", 0)
        if code_score > 0 and arena_score > 0:
            return (code_score * 0.6 + arena_score * 0.4)
        if code_score > 0:
            return code_score
        return 35.0

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
    return 40.0


# ── Semantic keyword expansion ───────────────────────────────────────────────
# Maps user query terms to related terms commonly found in model metadata.
# This bridges the vocabulary gap between how users describe needs vs how
# model capabilities are phrased in strengths/use_cases.
SEMANTIC_EXPANSIONS: dict[str, list[str]] = {
    "translate": ["multilingual", "language", "localize", "localization", "translation", "multi-lingual"],
    "translation": ["multilingual", "language", "localize", "multi-lingual"],
    "multilingual": ["translation", "language", "localize", "global"],
    "code": ["coding", "programming", "software", "developer", "engineering", "refactoring", "code generation"],
    "coding": ["code", "programming", "software", "developer", "engineering", "code generation"],
    "programming": ["code", "coding", "software", "developer", "software engineering"],
    "debug": ["debugging", "code", "issue", "bug", "error"],
    "python": ["code", "coding", "programming", "software", "developer"],
    "chatbot": ["chat", "conversation", "customer", "support", "assistant", "dialogue"],
    "chat": ["chatbot", "conversation", "customer", "assistant", "dialogue"],
    "customer service": ["chatbot", "support", "help desk", "customer"],
    "assistant": ["chatbot", "personal", "conversation", "real-time"],
    "summarize": ["summary", "document", "research", "analysis", "content"],
    "document": ["summarize", "research", "paper", "analysis", "long"],
    "legal": ["document", "contract", "regulation", "compliance", "research"],
    "research": ["analysis", "scientific", "paper", "data", "deep"],
    "science": ["scientific", "research", "math", "reasoning", "stem"],
    "math": ["mathematics", "numerical", "calculation", "equation", "quantitative", "stem"],
    "image": ["vision", "visual", "picture", "photo", "multimodal", "image processing"],
    "vision": ["image", "visual", "multimodal", "ocr", "image processing"],
    "ocr": ["image", "vision", "scan", "document", "text extraction"],
    "creative": ["writing", "narrative", "story", "fiction", "prose", "storytelling"],
    "writing": ["creative", "content", "narrative", "story", "prose"],
    "story": ["creative", "fiction", "narrative", "storytelling", "writing"],
    "agent": ["agentic", "autonomous", "automation", "workflow", "orchestration", "tool use"],
    "autonomous": ["agent", "agentic", "automation", "self-correction"],
    "fast": ["speed", "quick", "latency", "real-time", "efficient", "responsive"],
    "cheap": ["budget", "affordable", "cost-effective", "economical", "low-cost"],
    "budget": ["cheap", "affordable", "cost-effective", "economical"],
}


def _expand_query(use_case_lower: str) -> str:
    """Expand the user's query with semantically related terms.

    Adds related words that are commonly found in model metadata,
    bridging the vocabulary gap between user language and model descriptions.
    """
    expanded_terms = set()
    for trigger, expansions in SEMANTIC_EXPANSIONS.items():
        if trigger in use_case_lower:
            expanded_terms.update(expansions)

    if expanded_terms:
        return use_case_lower + " " + " ".join(expanded_terms)
    return use_case_lower


def _use_case_relevance(model, use_case_lower: str) -> float:
    """Dynamically score how well a model's curated metadata matches the use case.

    Matches the user's text (expanded with semantic synonyms) against the model's:
    - use_cases: what the model is designed for
    - strengths: what the model excels at
    - description: general model description

    No hardcoded model lists — purely data-driven from OpenRouter metadata
    and our curated enrichments.
    """
    # Expand the query with semantically related terms
    expanded_query = _expand_query(use_case_lower)
    total_bonus = 0.0

    # 1. Match against model's curated use_cases (highest signal, up to +14)
    use_case_score = 0.0
    for uc in model.use_cases:
        match = _phrase_match_score(expanded_query, uc)
        use_case_score += match
    total_bonus += min(use_case_score, 14.0)

    # 2. Match against model's strengths (up to +10)
    strength_score = 0.0
    for strength in model.strengths:
        match = _phrase_match_score(expanded_query, strength)
        strength_score += match
    total_bonus += min(strength_score, 10.0)

    # 3. Match against model's description (lighter weight, up to +4)
    if model.description:
        desc_match = _phrase_match_score(expanded_query, model.description)
        total_bonus += min(desc_match * 0.5, 4.0)

    # 4. Bonus if model's update_history mentions relevant terms (up to +3)
    history_score = 0.0
    for update in model.update_history:
        match = _phrase_match_score(expanded_query, update)
        history_score += match * 0.3
    total_bonus += min(history_score, 3.0)

    return min(total_bonus, 30.0)


def _generate_reasoning(model, dimension_scores: dict, priorities: dict,
                        relevance_bonus: float) -> str:
    """Generate a human-readable reasoning string."""
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
    if relevance_bonus >= 12:
        parts.insert(0, "Specialist match for your use case")
    elif relevance_bonus >= 6:
        parts.insert(0, "Strong alignment with your requirements")

    # Benchmark citation — pick the most relevant one
    if model.benchmarks:
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

    # All scoring dimensions — undetected ones get a low default
    # Cost and speed only get 0.5 unless explicitly requested, to prevent
    # cheap models from dominating niche queries
    all_dims = ["quality", "cost", "speed", "code", "reasoning", "math",
                "creative", "long_context", "multilingual", "vision", "agent", "chat"]
    LOW_DEFAULT_DIMS = {"cost", "speed"}  # these inflate scores for cheap models
    for dim in all_dims:
        if dim in LOW_DEFAULT_DIMS:
            priorities.setdefault(dim, 0.5)
        else:
            priorities.setdefault(dim, 1.0)

    use_case_lower = request.use_case.lower()
    scored_models: list[ModelScore] = []

    for model in models:
        # Apply hard filters
        if request.min_context_window and model.context_window < request.min_context_window:
            continue
        if request.budget_per_1m_tokens and model.input_price_per_1m > request.budget_per_1m_tokens:
            continue

        # Score each dimension via benchmarks + heuristics
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

        # Dynamic use-case relevance bonus — matches model metadata against user text
        relevance = _use_case_relevance(model, use_case_lower)
        overall = min(base_score + relevance, 100.0)

        reasoning = _generate_reasoning(model, dimension_scores, priorities, relevance)
        cost_est = _estimate_cost(model, request.use_case)

        # Show the most relevant dimension scores (top 6 by priority weight)
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
