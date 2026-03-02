"""WhichModel data layer — fetches live model data from OpenRouter API."""
from __future__ import annotations
from typing import Optional, List
from models import LLMModel
from openrouter import fetch_models


def get_all_models() -> List[LLMModel]:
    """Return all models (live from OpenRouter, cached)."""
    return fetch_models()


def get_model_by_id(model_id: str) -> Optional[LLMModel]:
    """Return a model by its slug ID."""
    for model in get_all_models():
        if model.id == model_id:
            return model
    return None


def get_models_by_ids(model_ids: List[str]) -> List[LLMModel]:
    """Return models matching the given IDs."""
    id_set = set(model_ids)
    return [m for m in get_all_models() if m.id in id_set]
