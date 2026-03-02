"""Model listing and comparison endpoints."""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from data import get_all_models, get_model_by_id, get_models_by_ids

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider name"),
    sort_by: Optional[str] = Query(None, description="Sort by: price_asc, price_desc, context, name"),
    search: Optional[str] = Query(None, description="Search by model name or description"),
    modality: Optional[str] = Query(None, description="Filter by modality: text, multimodal, image"),
    has_benchmarks: Optional[bool] = Query(None, description="Only show models with benchmark data"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=200, description="Models per page"),
):
    """List all available models with optional filtering, sorting, and pagination."""
    models = get_all_models()

    # Text search
    if search:
        q = search.lower()
        models = [m for m in models if q in m.name.lower() or q in m.description.lower() or q in m.provider.lower()]

    # Provider filter
    if provider:
        models = [m for m in models if m.provider.lower() == provider.lower()]

    # Modality filter
    if modality:
        mod = modality.lower()
        if mod == "multimodal":
            models = [m for m in models if "image" in m.input_modalities or "video" in m.input_modalities]
        elif mod == "text":
            models = [m for m in models if m.input_modalities == ["text"]]
        elif mod == "image":
            models = [m for m in models if "image" in m.output_modalities]

    # Benchmark filter
    if has_benchmarks:
        models = [m for m in models if len(m.benchmarks) > 0]

    # Sorting
    if sort_by == "price_asc":
        models.sort(key=lambda m: m.input_price_per_1m)
    elif sort_by == "price_desc":
        models.sort(key=lambda m: m.input_price_per_1m, reverse=True)
    elif sort_by == "context":
        models.sort(key=lambda m: m.context_window, reverse=True)
    elif sort_by == "name":
        models.sort(key=lambda m: m.name)

    # Pagination
    total = len(models)
    start = (page - 1) * per_page
    end = start + per_page
    page_models = models[start:end]

    # Collect unique providers for filter chips
    all_providers = sorted(set(m.provider for m in get_all_models()))

    return {
        "models": page_models,
        "count": len(page_models),
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "providers": all_providers,
    }


@router.get("/compare")
def compare_models(
    ids: str = Query(..., description="Comma-separated model IDs to compare"),
):
    """Compare models side-by-side."""
    model_ids = [mid.strip() for mid in ids.split(",")]
    if len(model_ids) < 2:
        raise HTTPException(400, "Provide at least 2 model IDs to compare")
    if len(model_ids) > 6:
        raise HTTPException(400, "Maximum 6 models for comparison")

    models = get_models_by_ids(model_ids)
    if len(models) < 2:
        raise HTTPException(404, "Some model IDs were not found")

    # Compute highlights — best/worst per metric
    all_benchmarks = set()
    for m in models:
        all_benchmarks.update(m.benchmarks.keys())

    benchmark_leaders = {}
    for bm in all_benchmarks:
        scores = [(m.id, m.benchmarks.get(bm, 0)) for m in models]
        scores.sort(key=lambda x: x[1], reverse=True)
        benchmark_leaders[bm] = {"best": scores[0][0], "worst": scores[-1][0]}

    price_sorted = sorted(models, key=lambda m: m.input_price_per_1m)
    context_sorted = sorted(models, key=lambda m: m.context_window, reverse=True)

    return {
        "models": models,
        "highlights": {
            "cheapest": price_sorted[0].id,
            "most_expensive": price_sorted[-1].id,
            "largest_context": context_sorted[0].id,
            "benchmark_leaders": benchmark_leaders,
        },
    }


@router.get("/{model_id}")
def get_model(model_id: str):
    """Get detailed info for a single model."""
    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(404, f"Model '{model_id}' not found")
    return model
