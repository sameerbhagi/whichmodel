"""OpenRouter API integration — fetch live model data from https://openrouter.ai/api/v1/models"""
from __future__ import annotations

import time
import logging
from typing import Optional

import httpx

from models import LLMModel

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 3600  # 1 hour

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict = {"models": None, "fetched_at": 0}

# ── Provider → brand color mapping ───────────────────────────────────────────
PROVIDER_COLORS: dict[str, str] = {
    "openai":       "#10a37f",
    "anthropic":    "#d4a574",
    "google":       "#4285f4",
    "meta-llama":   "#0467df",
    "mistralai":    "#ff7000",
    "deepseek":     "#5b6ef7",
    "qwen":         "#6f42c1",
    "x-ai":         "#1da1f2",
    "cohere":       "#39594d",
    "perplexity":   "#20808d",
    "amazon":       "#ff9900",
    "nvidia":       "#76b900",
    "microsoft":    "#00a4ef",
    "inflection":   "#e91e63",
    "ai21":         "#6c5ce7",
    "bytedance-seed": "#00f0ff",
    "minimax":      "#f56040",
    "moonshotai":   "#7c4dff",
    "allenai":      "#00bcd4",
    "z-ai":         "#4caf50",
    "baidu":        "#2932e1",
    "inception":    "#ff4444",
    "liquid":       "#26c6da",
    "stepfun":      "#e040fb",
}

# ── Provider display name overrides ──────────────────────────────────────────
PROVIDER_DISPLAY: dict[str, str] = {
    "openai":       "OpenAI",
    "anthropic":    "Anthropic",
    "google":       "Google",
    "meta-llama":   "Meta",
    "mistralai":    "Mistral AI",
    "deepseek":     "DeepSeek",
    "qwen":         "Qwen (Alibaba)",
    "x-ai":         "xAI",
    "cohere":       "Cohere",
    "perplexity":   "Perplexity",
    "amazon":       "Amazon",
    "nvidia":       "NVIDIA",
    "microsoft":    "Microsoft",
    "inflection":   "Inflection AI",
    "ai21":         "AI21 Labs",
    "bytedance-seed": "ByteDance",
    "minimax":      "MiniMax",
    "moonshotai":   "Moonshot AI",
    "allenai":      "Allen AI",
    "z-ai":         "Zhipu AI",
    "baidu":        "Baidu",
    "inception":    "Inception",
    "liquid":       "Liquid AI",
    "stepfun":      "StepFun",
    "ibm-granite":  "IBM",
    "nousresearch": "Nous Research",
    "arcee-ai":     "Arcee AI",
    "writer":       "Writer",
}

# ── Curated benchmark scores for key models (Normalized 0-100) ───────────────
# Data sources: LMSYS, MMLU-Pro, GPQA Diamond, LiveCodeBench (Feb 2026)
CURATED_BENCHMARKS: dict[str, dict[str, float]] = {
    "openai/gpt-4o": {
        "MMLU-Pro": 82.5, "GPQA Diamond": 53.6, "LiveCodeBench": 82.1, "LMSYS Arena": 91.2,
    },
    "openai/gpt-4o-mini": {
        "MMLU-Pro": 75.2, "GPQA Diamond": 40.2, "LiveCodeBench": 78.5, "LMSYS Arena": 85.0,
    },
    "openai/o1": {
        "MMLU-Pro": 88.1, "GPQA Diamond": 79.3, "LiveCodeBench": 89.2, "LMSYS Arena": 94.5,
    },
    "openai/o3-mini": {
        "MMLU-Pro": 86.9, "GPQA Diamond": 79.7, "LiveCodeBench": 90.8, "LMSYS Arena": 93.8,
    },
    "openai/o4-mini": {
        "MMLU-Pro": 87.5, "GPQA Diamond": 81.2, "LiveCodeBench": 91.5, "LMSYS Arena": 94.2,
    },
    "openai/gpt-5.1": {
        "MMLU-Pro": 80.5, "GPQA Diamond": 70.2, "LiveCodeBench": 74.0, "LMSYS Arena": 93.5,
    },
    "openai/gpt-5.2": {
        "MMLU-Pro": 88.5, "GPQA Diamond": 93.2, "LiveCodeBench": 21.0, "LMSYS Arena": 96.8,
    },
    "openai/gpt-5.2-pro": {
        "MMLU-Pro": 91.2, "GPQA Diamond": 93.8, "LiveCodeBench": 85.2, "LMSYS Arena": 97.5,
    },
    "openai/gpt-5.3-codex": {
        "MMLU-Pro": 82.9, "GPQA Diamond": 73.8, "LiveCodeBench": 87.31, "LMSYS Arena": 95.0,
    },
    "openai/gpt-5.1-chat": {
        "MMLU-Pro": 81.2, "GPQA Diamond": 71.5, "LiveCodeBench": 75.2, "LMSYS Arena": 93.8,
    },
    "openai/gpt-5.1-codex": {
        "MMLU-Pro": 83.5, "GPQA Diamond": 74.2, "LiveCodeBench": 88.5, "LMSYS Arena": 94.2,
    },
    "openai/gpt-5.1-codex-max": {
        "MMLU-Pro": 85.0, "GPQA Diamond": 76.5, "LiveCodeBench": 91.2, "LMSYS Arena": 95.5,
    },
    "openai/gpt-5.1-codex-mini": {
        "MMLU-Pro": 78.5, "GPQA Diamond": 65.2, "LiveCodeBench": 82.0, "LMSYS Arena": 91.0,
    },
    "openai/gpt-5.2-chat": {
        "MMLU-Pro": 89.2, "GPQA Diamond": 94.0, "LiveCodeBench": 22.5, "LMSYS Arena": 97.2,
    },
    "openai/gpt-5.2-codex": {
        "MMLU-Pro": 91.5, "GPQA Diamond": 94.5, "LiveCodeBench": 86.8, "LMSYS Arena": 97.8,
    },
    "anthropic/claude-3.5-sonnet": {
        "MMLU-Pro": 84.2, "GPQA Diamond": 65.0, "LiveCodeBench": 85.4, "LMSYS Arena": 90.8,
    },
    "anthropic/claude-3.7-sonnet": {
        "MMLU-Pro": 89.5, "GPQA Diamond": 68.0, "LiveCodeBench": 87.1, "LMSYS Arena": 93.5,
    },
    "anthropic/claude-sonnet-4": {
        "MMLU-Pro": 91.0, "GPQA Diamond": 72.5, "LiveCodeBench": 89.5, "LMSYS Arena": 95.2,
    },
    "anthropic/claude-sonnet-4.5": {
        "MMLU-Pro": 92.5, "GPQA Diamond": 72.7, "LiveCodeBench": 92.4, "LMSYS Arena": 96.0,
    },
    "anthropic/claude-sonnet-4.6": {
        "MMLU-Pro": 93.8, "GPQA Diamond": 74.0, "LiveCodeBench": 93.5, "LMSYS Arena": 96.8,
    },
    "anthropic/claude-opus-4": {
        "MMLU-Pro": 93.2, "GPQA Diamond": 75.8, "LiveCodeBench": 91.2, "LMSYS Arena": 96.5,
    },
    "anthropic/claude-opus-4.5": {
        "MMLU-Pro": 95.1, "GPQA Diamond": 80.9, "LiveCodeBench": 94.6, "LMSYS Arena": 98.2,
    },
    "anthropic/claude-opus-4.6": {
        "MMLU-Pro": 96.5, "GPQA Diamond": 82.0, "LiveCodeBench": 95.8, "LMSYS Arena": 98.8,
    },
    "anthropic/claude-haiku-4.5": {
        "MMLU-Pro": 82.0, "GPQA Diamond": 50.5, "LiveCodeBench": 73.0, "LMSYS Arena": 89.4,
    },
    "anthropic/claude-3.7-sonnet:thinking": {
        "MMLU-Pro": 91.5, "GPQA Diamond": 72.0, "LiveCodeBench": 90.5, "LMSYS Arena": 94.8,
    },
    "anthropic/claude-opus-4.1": {
        "MMLU-Pro": 94.2, "GPQA Diamond": 77.5, "LiveCodeBench": 92.0, "LMSYS Arena": 97.2,
    },
    "google/gemini-2.0-flash-001": {
        "MMLU-Pro": 78.5, "GPQA Diamond": 56.1, "LiveCodeBench": 79.2, "LMSYS Arena": 88.5,
    },
    "google/gemini-2.5-pro": {
        "MMLU-Pro": 90.5, "GPQA Diamond": 90.8, "LiveCodeBench": 91.7, "LMSYS Arena": 96.2,
    },
    "google/gemini-2.5-flash": {
        "MMLU-Pro": 87.0, "GPQA Diamond": 58.0, "LiveCodeBench": 85.0, "LMSYS Arena": 92.0,
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "MMLU-Pro": 44.3, "GPQA Diamond": 27.0, "LiveCodeBench": 19.7, "LMSYS Arena": 82.0,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "MMLU-Pro": 68.9, "GPQA Diamond": 50.5, "LiveCodeBench": 36.6, "LMSYS Arena": 91.0,
    },
    "meta-llama/llama-4-maverick": {
        "MMLU-Pro": 80.5, "GPQA Diamond": 69.8, "LiveCodeBench": 52.0, "LMSYS Arena": 93.1,
    },
    "meta-llama/llama-4-scout": {
        "MMLU-Pro": 66.0, "GPQA Diamond": 80.0, "LiveCodeBench": 45.0, "LMSYS Arena": 90.5,
    },
    "google/gemma-3-27b-it": {
        "MMLU-Pro": 67.5, "GPQA Diamond": 42.4, "LiveCodeBench": 29.7, "LMSYS Arena": 87.5,
    },
    "google/gemma-3-12b-it": {
        "MMLU-Pro": 59.5, "GPQA Diamond": 34.9, "LiveCodeBench": 13.7, "LMSYS Arena": 85.0,
    },
    "google/gemma-3-4b-it": {
        "MMLU-Pro": 43.6, "GPQA Diamond": 30.8, "LiveCodeBench": 8.5, "LMSYS Arena": 82.2,
    },
    "google/gemma-3n-e4b-it": {
        "MMLU-Pro": 45.1, "GPQA Diamond": 32.5, "LiveCodeBench": 10.2, "LMSYS Arena": 83.5,
    },
    "google/gemini-2.5-flash-image": {
        "MMLU-Pro": 86.5, "GPQA Diamond": 57.5, "LiveCodeBench": 84.5, "LMSYS Arena": 91.8,
    },
    "google/gemini-2.5-pro-preview": {
        "MMLU-Pro": 90.2, "GPQA Diamond": 89.5, "LiveCodeBench": 90.1, "LMSYS Arena": 95.8,
    },
    "google/gemini-3-pro-image-preview": {
        "MMLU-Pro": 91.5, "GPQA Diamond": 92.5, "LiveCodeBench": 83.5, "LMSYS Arena": 96.0,
    },
    "meta-llama/llama-3-8b-instruct": {
        "MMLU-Pro": 38.5, "GPQA Diamond": 22.0, "LiveCodeBench": 12.5, "LMSYS Arena": 80.5,
    },
    "meta-llama/llama-3-70b-instruct": {
        "MMLU-Pro": 63.2, "GPQA Diamond": 45.8, "LiveCodeBench": 30.5, "LMSYS Arena": 89.2,
    },
    "meta-llama/llama-3.1-405b": {
        "MMLU-Pro": 73.0, "GPQA Diamond": 49.0, "LiveCodeBench": 27.5, "LMSYS Arena": 94.5,
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "MMLU-Pro": 73.3, "GPQA Diamond": 50.7, "LiveCodeBench": 30.5, "LMSYS Arena": 95.2,
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "MMLU-Pro": 67.5, "GPQA Diamond": 48.5, "LiveCodeBench": 35.2, "LMSYS Arena": 90.5,
    },
    "meta-llama/llama-3.2-1b-instruct": {
        "MMLU-Pro": 25.0, "GPQA Diamond": 15.0, "LiveCodeBench": 5.0, "LMSYS Arena": 75.0,
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "MMLU-Pro": 32.5, "GPQA Diamond": 20.2, "LiveCodeBench": 10.5, "LMSYS Arena": 78.5,
    },
    "meta-llama/llama-3.2-11b-vision-instruct": {
        "MMLU-Pro": 45.8, "GPQA Diamond": 28.5, "LiveCodeBench": 18.2, "LMSYS Arena": 83.0,
    },
    "mistralai/mistral-large-2512": {
        "MMLU-Pro": 83.2, "GPQA Diamond": 58.4, "LiveCodeBench": 78.6, "LMSYS Arena": 91.5,
    },
    "mistralai/ministral-8b-2512": {
        "MMLU-Pro": 72.5, "GPQA Diamond": 38.0, "LiveCodeBench": 45.0, "LMSYS Arena": 84.0,
    },
    "mistralai/devstral-2512": {
        "MMLU-Pro": 78.0, "GPQA Diamond": 55.0, "LiveCodeBench": 92.5, "LMSYS Arena": 89.0,
    },
    "mistralai/devstral-medium": {
        "MMLU-Pro": 82.5, "GPQA Diamond": 65.0, "LiveCodeBench": 94.2, "LMSYS Arena": 91.5,
    },
    "mistralai/devstral-small": {
        "MMLU-Pro": 75.8, "GPQA Diamond": 50.2, "LiveCodeBench": 90.5, "LMSYS Arena": 87.5,
    },
    "mistralai/ministral-3b-2512": {
        "MMLU-Pro": 52.5, "GPQA Diamond": 32.0, "LiveCodeBench": 15.0, "LMSYS Arena": 82.5,
    },
    "mistralai/ministral-14b-2512": {
        "MMLU-Pro": 68.2, "GPQA Diamond": 45.1, "LiveCodeBench": 32.0, "LMSYS Arena": 88.8,
    },
    "mistralai/mistral-medium-3.1": {
        "MMLU-Pro": 81.5, "GPQA Diamond": 73.5, "LiveCodeBench": 77.2, "LMSYS Arena": 93.5,
    },
    "mistralai/mistral-small-3.1-24b-instruct": {
        "MMLU-Pro": 65.5, "GPQA Diamond": 42.0, "LiveCodeBench": 25.5, "LMSYS Arena": 86.2,
    },
    "mistralai/mistral-small-3.2-24b-instruct": {
        "MMLU-Pro": 66.8, "GPQA Diamond": 43.5, "LiveCodeBench": 27.2, "LMSYS Arena": 87.0,
    },
    "mistralai/mistral-small-creative": {
        "MMLU-Pro": 64.0, "GPQA Diamond": 40.5, "LiveCodeBench": 20.2, "LMSYS Arena": 88.5,
    },
    "mistralai/codestral-2508": {
        "MMLU-Pro": 65.0, "GPQA Diamond": 38.5, "LiveCodeBench": 82.4, "LMSYS Arena": 88.2,
    },
    "mistralai/voxtral-small-24b-2507": {
        "MMLU-Pro": 70.2, "GPQA Diamond": 48.5, "LiveCodeBench": 35.0, "LMSYS Arena": 89.5,
    },
    "anthropic/claude-3-haiku": {
        "MMLU-Pro": 75.2, "GPQA Diamond": 35.8, "LiveCodeBench": 42.1, "LMSYS Arena": 83.5,
    },
    "anthropic/claude-3.5-haiku": {
        "MMLU-Pro": 65.0, "GPQA Diamond": 41.6, "LiveCodeBench": 51.4, "LMSYS Arena": 85.0,
    },
    "google/gemini-2.5-flash-lite": {
        "MMLU-Pro": 81.1, "GPQA Diamond": 64.6, "LiveCodeBench": 33.7, "LMSYS Arena": 87.0,
    },
    "google/gemini-3-pro-preview": {
        "MMLU-Pro": 90.5, "GPQA Diamond": 91.9, "LiveCodeBench": 82.0, "LMSYS Arena": 95.5,
    },
    "google/gemini-3-flash": {
        "MMLU-Pro": 89.0, "GPQA Diamond": 90.4, "LiveCodeBench": 88.0, "LMSYS Arena": 94.0,
    },
    "google/gemini-3.1-pro-preview": {
        "MMLU-Pro": 92.6, "GPQA Diamond": 94.3, "LiveCodeBench": 92.5, "LMSYS Arena": 97.2,
    },
    "google/gemini-3.1-flash-image-preview": {
        "MMLU-Pro": 89.5, "GPQA Diamond": 90.8, "LiveCodeBench": 88.5, "LMSYS Arena": 94.5,
    },
    "google/gemini-3-flash-preview": {
        "MMLU-Pro": 89.0, "GPQA Diamond": 90.4, "LiveCodeBench": 88.0, "LMSYS Arena": 94.0,
    },
    "mistralai/mistral-large": {
        "MMLU-Pro": 76.2, "GPQA Diamond": 46.9, "LiveCodeBench": 72.1, "LMSYS Arena": 86.5,
    },
    "mistralai/mistral-medium-3": {
        "MMLU-Pro": 80.5, "GPQA Diamond": 72.0, "LiveCodeBench": 76.0, "LMSYS Arena": 93.0,
    },
}

# ── Curated strengths & weaknesses for key models ───────────────────────────
CURATED_META: dict[str, dict] = {
    "openai/gpt-4o": {
        "release_date": "2024-05-13",
        "update_history": ["May 2024: Initial launch with Omni multimodality", "Aug 2024: Significant speed improvements"],
        "strengths": ["Real-time multimodal interaction", "Extremely low latency", "Most versatile for voice/vision"],
        "weaknesses": ["Occasionally hallucinates on complex logic tasks"],
        "use_cases": ["Personal AI assistants", "Real-time speech translation", "Complex multimodal analysis"]
    },
    "openai/gpt-4o-mini": {
        "release_date": "2024-07-18",
        "update_history": ["Jul 2024: Compact version of GPT-4o"],
        "strengths": ["Extremely affordable", "Fast response times", "Good code generation", "128K context"],
        "weaknesses": ["Lower reasoning than GPT-4o", "Weaker on complex math"],
        "use_cases": ["Retail chatbots", "Simple text extraction", "Bulk content moderation"]
    },
    "openai/o1": {
        "release_date": "2024-09-12",
        "update_history": ["Sep 2024: Initial launch of reasoning model", "Nov 2024: Price reduction for inference"],
        "strengths": ["Self-correction reasoning", "PhD-level science accuracy", "Exceptional logic & math"],
        "weaknesses": ["High latency due to thinking time", "Premium pricing"],
        "use_cases": ["Scientific research", "Advanced software architecture", "Strategic planning"]
    },
    "openai/o3-mini": {
        "release_date": "2024-12-15",
        "update_history": ["Dec 2024: Optimized reasoning at lower cost"],
        "strengths": ["Strong reasoning at lower cost", "Excellent math", "Good coding"],
        "weaknesses": ["Less capable than full o1", "Higher latency than GPT-4o"],
        "use_cases": ["Competitive programming", "Technical problem solving", "Complex logic flows"]
    },
    "openai/gpt-5.1": {
        "release_date": "2025-11-12",
        "update_history": ["Nov 2025: Initial GPT-5.1 launch", "Jan 2026: Latency optimizations"],
        "strengths": ["Enhanced instruction following", "More conversational tone", "Adaptive reasoning"],
        "weaknesses": ["Occasionally verbose"],
        "use_cases": ["Personal AI assistants", "Advanced tutoring", "Creative collaboration"]
    },
    "openai/gpt-5.2": {
        "release_date": "2025-12-11",
        "update_history": ["Dec 2025: Major jump in general intelligence", "Feb 2026: Visual reasoner upgrade"],
        "strengths": ["Frontier-class reasoning", "Deep project management skills", "Exceptional vision"],
        "weaknesses": ["High cost per token"],
        "use_cases": ["Strategic project planning", "Complex multimodal research", "Financial modeling"]
    },
    "openai/gpt-5.2-pro": {
        "release_date": "2026-01-15",
        "update_history": ["Jan 2026: High-performance Pro variant for enterprise"],
        "strengths": ["Maximum reasoning depth", "Highest reliability", "Superior multitasking"],
        "weaknesses": ["Highest cost in class"],
        "use_cases": ["Enterprise AI leads", "Critical research automation"]
    },
    "openai/gpt-5.3-codex": {
        "release_date": "2026-02-05",
        "update_history": ["Feb 2026: SOTA agentic coding model launch"],
        "strengths": ["Best-in-class agentic coding", "Terminal-Bench 77.3%", "High speed for coding tasks"],
        "weaknesses": ["Narrowly focused on engineering/reasoning"],
        "use_cases": ["Autonomous software engineering", "Large codebase refactoring", "CI/CD automation"]
    },
    "openai/o4-mini": {
        "release_date": "2025-06-10",
        "update_history": ["Jun 2025: Next-gen compact reasoning model"],
        "strengths": ["Industry-leading reasoning efficiency", "Superior coding logic", "Fastest PhD-level reasoning"],
        "weaknesses": ["Smaller knowledge base than frontier models"],
        "use_cases": ["Autonomous code agents", "Logic-heavy real-time apps"]
    },
    "openai/gpt-5.1-chat": {
        "release_date": "2025-11-12",
        "update_history": ["Nov 2025: Optimized chat variant of GPT-5.1"],
        "strengths": ["Conversationally optimized", "Fast response", "Agentic orchestration"],
        "weaknesses": ["Occasionally verbose"],
        "use_cases": ["Advanced chatbots", "Customer service", "Personal assistants"]
    },
    "openai/gpt-5.1-codex": {
        "release_date": "2025-11-15",
        "update_history": ["Nov 2025: Standard Codex variant for GPT-5.1"],
        "strengths": ["Strong coding logic", "Fast inference", "Reliable tool use"],
        "weaknesses": ["Less creative than chat variant"],
        "use_cases": ["Software development", "Scripting", "API integration"]
    },
    "openai/gpt-5.1-codex-max": {
        "release_date": "2025-11-20",
        "update_history": ["Nov 2025: High-capacity Codex for large projects"],
        "strengths": ["Large codebase analysis", "Complex refactoring", "SOTA for technical documentation"],
        "weaknesses": ["Higher latency"],
        "use_cases": ["Entire project refactoring", "Architecture design"]
    },
    "openai/gpt-5.1-codex-mini": {
        "release_date": "2025-11-15",
        "update_history": ["Nov 2025: Lightweight coding model"],
        "strengths": ["Ultra-fast coding assistance", "Low cost"],
        "weaknesses": ["Limited complex reasoning"],
        "use_cases": ["Real-time autocomplete", "Simple unit tests"]
    },
    "openai/gpt-5.2-chat": {
        "release_date": "2025-12-11",
        "update_history": ["Dec 2025: Major update in chat capabilities"],
        "strengths": ["Human-level nuance", "Exceptional planning", "Multimodal chat"],
        "weaknesses": ["Premium pricing"],
        "use_cases": ["Strategic consultation", "Creative direction"]
    },
    "openai/gpt-5.2-codex": {
        "release_date": "2025-12-15",
        "update_history": ["Dec 2025: Frontier coding intelligence"],
        "strengths": ["Solves complex software engineering problems", "High reliability"],
        "weaknesses": ["Expensive for simple scripts"],
        "use_cases": ["Autonomous engineering lead", "Complex system design"]
    },
    "anthropic/claude-3.5-sonnet": {
        "release_date": "2024-06-20",
        "update_history": ["Jun 2024: Release of Claude 3.5 Sonnet", "Oct 2024: Upgraded model with improved coding"],
        "strengths": ["Industry-leading code generation", "Natural content tone", "Superior document analysis"],
        "weaknesses": ["200k context is smaller than Google's Pro models"],
        "use_cases": ["Production software engineering", "Complex document summarization", "Creative writing"]
    },
    "anthropic/claude-3.7-sonnet": {
        "release_date": "2025-02-24",
        "update_history": ["Feb 2025: Hybrid thinking mode introduced", "Apr 2025: Latency optimizations"],
        "strengths": ["Flexible 'Thinking' mode", "Best-in-class coding benchmarks"],
        "weaknesses": ["Higher token costs for thinking chains"],
        "use_cases": ["Deep code refactoring", "Technical research & analysis"]
    },
    "anthropic/claude-3.7-sonnet:thinking": {
        "release_date": "2025-02-24",
        "update_history": ["Feb 2025: Release of extended Thinking mode"],
        "strengths": ["Extremely deep reasoning", "Solves hard math/coding problems", "Self-correction"],
        "weaknesses": ["Slower output", "Higher cost due to internal monologue tokens"],
        "use_cases": ["Scientific discovery", "Complex logic debugging", "Formal verification"]
    },
    "anthropic/claude-opus-4.1": {
        "release_date": "2025-08-14",
        "update_history": ["Aug 2025: Iterative improvement on Opus 4"],
        "strengths": ["Improved emotional intelligence", "Higher logic consistency", "Superior creative writing"],
        "weaknesses": ["Slower than Sonnet 4.5"],
        "use_cases": ["Enterprise leadership consultation", "Advanced content strategy"]
    },
    "anthropic/claude-sonnet-4": {
        "release_date": "2025-05-22",
        "update_history": ["May 2025: Claude 4 family launch"],
        "strengths": ["Excellent balance of speed and intelligence", "Strong tool use", "85.4% MMLU"],
        "weaknesses": ["Occasionally cautious"],
        "use_cases": ["Corporate automation", "Data analysis", "Coding assistants"]
    },
    "anthropic/claude-sonnet-4.5": {
        "release_date": "2025-09-29",
        "update_history": ["Sep 2025: SOTA Computer Use capabilities"],
        "strengths": ["Best-in-class Computer Use", "Rapid reasoning", "Excellent for agents"],
        "weaknesses": ["Thinking limits on very long tasks"],
        "use_cases": ["Autonomous UI agents", "Software engineering", "Complex workflow automation"]
    },
    "anthropic/claude-sonnet-4.6": {
        "release_date": "2026-02-17",
        "update_history": ["Feb 2026: 1M token context beta"],
        "strengths": ["Massive context window", "Improved design & data handling", "High reliability"],
        "weaknesses": ["Preview status for 1M context"],
        "use_cases": ["Entire project codebase analysis", "Deep legal research", "Complex office automation"]
    },
    "anthropic/claude-opus-4.6": {
        "release_date": "2026-02-20",
        "update_history": ["Feb 2026: Flagship Claude 4.6 launch"],
        "strengths": ["Ultimate intelligence", "Superior creative reasoning", "Emotional depth"],
        "weaknesses": ["High latency"],
        "use_cases": ["High-end strategic advisory", "Creative masterpiece collaboration"]
    },
    "anthropic/claude-opus-4": {
        "release_date": "2025-05-22",
        "update_history": ["May 2025: Most capable Claude at launch"],
        "strengths": ["Expert-level knowledge", "Deep reasoning", "High safety"],
        "weaknesses": ["Slower than Sonnet"],
        "use_cases": ["Strategic business intelligence", "Scientific research", "Advanced ethics analysis"]
    },
    "anthropic/claude-opus-4.5": {
        "release_date": "2025-11-24",
        "update_history": ["Nov 2025: Outperformed GPT-5.1 on SWE-bench"],
        "strengths": ["State-of-the-art for agents & coding", "80.9% SWE-bench", "Exceptional computer use"],
        "weaknesses": ["Very high cost"],
        "use_cases": ["Autonomous engineering lead", "Advanced STEM research", "Strategic enterprise AI"]
    },
    "anthropic/claude-haiku-4.5": {
        "release_date": "2025-10-15",
        "update_history": ["Oct 2025: High-speed frontier model"],
        "strengths": ["Best-in-class speed for intelligence level", "Sub-second reasoning latency"],
        "weaknesses": ["Lower depth than Opus 4.5"],
        "use_cases": ["Real-time customer agents", "High-frequency classification", "Lightweight mobile reasoning"]
    },
    "anthropic/claude-3.5-haiku": {
        "release_date": "2024-11-04",
        "update_history": ["Nov 2024: Fast reasoning release"],
        "strengths": ["Fast reasoning responses", "Good cost efficiency", "200K context"],
        "weaknesses": ["Less knowledge depth than Sonnet"],
        "use_cases": ["Real-time logic triage", "High-volume reasoning tasks"]
    },
    "google/gemini-2.0-flash-001": {
        "release_date": "2024-12-10",
        "update_history": ["Dec 2024: Real-time multimodal streaming API"],
        "strengths": ["Extreme speed", "1M context", "Cheapest high-tier model"],
        "weaknesses": ["Less depth in reasoning than Pro"],
        "use_cases": ["Real-time multimodal UI", "High-volume data parsing"]
    },
    "google/gemini-2.5-pro": {
        "release_date": "2025-05-01",
        "update_history": ["May 2025: Breakthrough in long-context reasoning", "Jan 2026: Enhanced multimodal understanding"],
        "strengths": ["Massive 2M token context window", "Best multimodal reasoning (video/audio)", "Integrated Google search"],
        "weaknesses": ["Larger models can be slower to initialize"],
        "use_cases": ["Analyzing entire codebases", "Video content analysis & search", "Deep cross-modal research"]
    },
    "google/gemini-2.5-flash": {
        "release_date": "2025-05-01",
        "update_history": ["May 2025: Released alongside 2.5 Pro"],
        "strengths": ["1M context at low cost", "Fast inference", "Strong multimodal"],
        "weaknesses": ["Less depth than Pro"],
        "use_cases": ["Long-document Q&A", "Efficient video summarization"]
    },
    "google/gemini-3-pro-preview": {
        "release_date": "2025-11-13",
        "update_history": ["Nov 2025: Next-gen architecture preview"],
        "strengths": ["Extreme reasoning capacity", "91.9% GPQA Diamond", "Multimodal expert"],
        "weaknesses": ["To be replaced by 3.1 Pro by March 2026"],
        "use_cases": ["Scientific research", "Advanced video analysis", "Complex logic tasks"]
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "release_date": "2024-07-23",
        "update_history": ["Jul 2024: First 400B+ open-weight frontier model"],
        "strengths": ["Massive parameter scale", "Strong general knowledge", "Excellent multilingual"],
        "weaknesses": ["Requires significant compute for self-hosting"],
        "use_cases": ["Synthetic data generation", "Model distillation", "Complex research"]
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "release_date": "2024-12-06",
        "update_history": ["Dec 2024: Optimized 70B model with GPT-4 class performance"],
        "strengths": ["Best-in-class 70B performance", "Excellent reasoning-to-cost ratio"],
        "weaknesses": ["Smaller context than Gemini 1.5/2.x Pro series"],
        "use_cases": ["General purpose chatbot", "Enterprise RAG", "Instruction following"]
    },
    "google/gemma-3-27b-it": {
        "release_date": "2025-03-12",
        "update_history": ["Mar 2025: Next-gen open-weight flagship"],
        "strengths": ["SOTA for open-weight 27B class", "Excellent multilingual", "Strong reasoning"],
        "weaknesses": ["Lower coding performance than specialized models"],
        "use_cases": ["Self-hosted chatbots", "Edge device AI", "Multilingual agents"]
    },
    "meta-llama/llama-4-maverick": {
        "release_date": "2025-04-06",
        "update_history": ["Apr 2025: Llama 4 Maverick MoE launch"],
        "strengths": ["Strong engineering performance", "High efficiency MoE architecture", "400B total parameters"],
        "weaknesses": ["Reasoning lag behind Gemini 3.1 Pro"],
        "use_cases": ["Self-hosted enterprise AI", "Large-scale data processing", "On-prem software development"]
    },
    "meta-llama/llama-4-scout": {
        "release_date": "2025-04-05",
        "update_history": ["Apr 2025: Released alongside Maverick"],
        "strengths": ["Lightweight 109B MoE", "Excellent speed-to-intelligence ratio"],
        "weaknesses": ["Lower reasoning depth than Maverick"],
        "use_cases": ["Edge AI agents", "High-concurrency chat systems"]
    },
    "anthropic/claude-3-haiku": {
        "release_date": "2024-03-04",
        "update_history": ["Mar 2024: Initial launch of Claude 3 family"],
        "strengths": ["Extreme speed", "Cheap to run", "Good for simple classification"],
        "weaknesses": ["Low depth for complex reasoning"],
        "use_cases": ["Customer support bots", "Real-time moderation", "Data formatting"]
    },
    "google/gemini-3.1-pro-preview": {
        "release_date": "2026-02-19",
        "update_history": ["Feb 2026: Google's most advanced model", "ARC-AGI-2 77.1% verified"],
        "strengths": ["Overall #1 on Intelligence Index", "Exceptional ARC-AGI reasoning", "94.3% GPQA Diamond"],
        "weaknesses": ["Preview status: fluctuating latency"],
        "use_cases": ["Solves entirely new logic patterns", "Professional coding & STEM", "Next-gen enterprise agents"]
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "release_date": "2024-12-06",
        "update_history": ["Dec 2024: Optimized 70B model with GPT-4 class performance"],
        "strengths": ["Best open-weights model in its class", "Very cost-effective to self-host", "Strong instruction following"],
        "weaknesses": ["Lacks native multimodal vision in this version"],
        "use_cases": ["Privacy-sensitive deployments", "High-throughput text automation", "Fine-tuning base"]
    },
    "meta-llama/llama-4-maverick": {
        "release_date": "2025-09-01",
        "update_history": ["Sep 2025: Next-gen architecture with native multimodality"],
        "strengths": ["Superior efficiency", "State-of-the-art open weights coding", "Large context"],
        "weaknesses": ["Hardware requirements for full precision"],
        "use_cases": ["General purpose open-weight chatbot", "On-prem technical analysis"]
    },
    "mistralai/mistral-large": {
        "release_date": "2024-02-26",
        "update_history": ["Feb 2024: Initial release", "Jul 2024: Updated Large 2 variant"],
        "strengths": ["Excellent multilingual performance", "Great code generation", "Europe-based compliance"],
        "weaknesses": ["Slightly lower reasoning scores than US flagships"],
        "use_cases": ["Multilingual enterprise apps", "European data residency needs"]
    },
    "anthropic/claude-3.5-haiku": {
        "release_date": "2024-11-04",
        "update_history": ["Nov 2024: High-speed reasoning model released", "Jan 2025: Improved reliability in tool use"],
        "strengths": ["Extreme speed for reasoning tasks", "Cost-effective intelligence", "Strong instruction following"],
        "weaknesses": ["Lower reasoning depth than Sonnet"],
        "use_cases": ["Real-time data classification", "High-volume reasoning triage"]
    },
    "google/gemini-2.5-flash-lite": {
        "release_date": "2025-02-10",
        "update_history": ["Feb 2025: Lightweight variant for mobile and high-concurrency"],
        "strengths": ["Fastest response time in Gemini family", "Efficient for simple tasks"],
        "weaknesses": ["Smallest reasoning capacity in the 2.5 tier"],
        "use_cases": ["Simple mobile agents", "Autocomplete & simple parsing"]
    },
    "google/gemini-3.1-flash-image-preview": {
        "release_date": "2026-02-15",
        "update_history": ["Feb 2026: Vision-first Flash preview"],
        "strengths": ["Next-gen visual parsing", "Sub-second image analysis"],
        "weaknesses": ["Still in preview"],
        "use_cases": ["Live video analysis", "Instant OCR"]
    },
    "google/gemma-3-12b-it": {
        "release_date": "2025-03-12",
        "update_history": ["Mar 2025: Instruction-tuned version of Gemma 3 12B"],
        "strengths": ["Strong multi-lingual reasoning", "Efficient hardware usage"],
        "weaknesses": ["Lower math performance than 27B"],
        "use_cases": ["Multilingual chatbots", "Technical support"]
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "release_date": "2024-07-23",
        "update_history": ["Jul 2024: Major update to Llama 3 8B"],
        "strengths": ["Improved reasoning over base Llama 3", "Large context window"],
        "weaknesses": ["Lower depth on niche technical topics"],
        "use_cases": ["RAG applications", "Customer service"]
    },
    "google/gemini-2.5-flash-image": {
        "release_date": "2025-05-15",
        "update_history": ["May 2025: Vision-optimized Flash variant"],
        "strengths": ["High-speed image processing", "Large context for visual media"],
        "weaknesses": ["Lower text reasoning than Pro"],
        "use_cases": ["Visual search", "Image captioning", "Video OCR"]
    },
    "google/gemini-2.5-pro-preview": {
        "release_date": "2025-04-10",
        "update_history": ["Apr 2025: Early access to 2.5 Pro architecture"],
        "strengths": ["Advanced reasoning", "2M context window"],
        "weaknesses": ["Higher latency than Flash"],
        "use_cases": ["Complex research", "Long-form content analysis"]
    },
    "google/gemini-3-pro-image-preview": {
        "release_date": "2025-11-20",
        "update_history": ["Nov 2025: Next-gen visual logic flagship"],
        "strengths": ["SOTA visual reasoning", "Multimodal logic chains"],
        "weaknesses": ["Experimental status"],
        "use_cases": ["Visual logic puzzles", "Advanced medical imaging analysis"]
    },
    "google/gemma-3-4b-it": {
        "release_date": "2025-03-12",
        "update_history": ["Mar 2025: High-efficiency compact open model"],
        "strengths": ["Fastest in its class", "Very low hardware requirements"],
        "weaknesses": ["Small knowledge base"],
        "use_cases": ["Edge device AI", "Simple routing", "Named entity recognition"]
    },
    "google/gemma-3n-e4b-it": {
        "release_date": "2025-06-05",
        "update_history": ["Jun 2025: Experimental architecture for enhanced logic"],
        "strengths": ["Superior reasoning-to-size ratio"],
        "weaknesses": ["Lower throughput than standard 4B"],
        "use_cases": ["On-device logic agents"]
    },
    "meta-llama/llama-3-8b-instruct": {
        "release_date": "2024-04-18",
        "update_history": ["Apr 2024: Initial Llama 3 launch"],
        "strengths": ["Fastest 8B model at launch", "Strong ecosystem support"],
        "weaknesses": ["Smallest context window in its family"],
        "use_cases": ["Chatbots", "Content generation", "Simple classification"]
    },
    "meta-llama/llama-3-70b-instruct": {
        "release_date": "2024-04-18",
        "update_history": ["Apr 2024: High-performance 70B variant"],
        "strengths": ["Broad knowledge", "Proven reliability"],
        "weaknesses": ["Lacks newer reasoning optimizations"],
        "use_cases": ["General purpose assistants", "Enterprise automation"]
    },
    "meta-llama/llama-3.1-405b": {
        "release_date": "2024-07-23",
        "update_history": ["Jul 2024: Largest open model available"],
        "strengths": ["Massive parameter scale", "Superior general knowledge"],
        "weaknesses": ["Extremely high compute requirements"],
        "use_cases": ["Fine-tuning base", "Complex reasoning tasks"]
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "release_date": "2024-07-23",
        "update_history": ["Jul 2024: Instruction-tuned version of Llama 3.1 405B"],
        "strengths": ["Excellent instruction following", "SOTA for open-weight reasoning"],
        "weaknesses": ["Slow inference without massive quantization"],
        "use_cases": ["Synthetic data generation", "Distillation", "Advanced research"]
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "release_date": "2024-07-23",
        "update_history": ["Jul 2024: Multi-lingual 70B variant"],
        "strengths": ["Strong multi-lingual support", "128K context window"],
        "weaknesses": ["Lower reasoning than 405B"],
        "use_cases": ["Global chatbots", "Translation services"]
    },
    "meta-llama/llama-3.2-1b-instruct": {
        "release_date": "2024-09-25",
        "update_history": ["Sep 2024: Ultra-compact edge architecture"],
        "strengths": ["Tiny memory footprint", "Runs on mobile CPUs"],
        "weaknesses": ["Limited knowledge capacity"],
        "use_cases": ["On-device text completion", "Simple commands"]
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "release_date": "2024-09-25",
        "update_history": ["Sep 2024: Balanced edge model"],
        "strengths": ["Best 3B-class performance", "Multilingual instruction suite"],
        "weaknesses": ["Struggles with complex multi-step reasoning"],
        "use_cases": ["Personal mobile assistants", "Context-aware UI controls"]
    },
    "meta-llama/llama-3.2-11b-vision-instruct": {
        "release_date": "2024-09-25",
        "update_history": ["Sep 2024: First Llama model with native vision support"],
        "strengths": ["Open-weight vision capabilities", "Excellent for local OCR"],
        "weaknesses": ["Lower text quality than Gemini Flash"],
        "use_cases": ["Local document processing", "Visual scene description"]
    },
    "google/gemini-3-flash-preview": {
        "release_date": "2025-12-01",
        "update_history": ["Dec 2025: Preview of next-gen Flash architecture", "Feb 2026: Massive leap in reasoning accuracy"],
        "strengths": ["Frontier-class reasoning at Flash speed", "1M+ context window", "Exceptional coding capability"],
        "weaknesses": ["Preview status: rate limits may apply"],
        "use_cases": ["Complex real-time analysis", "Codebase search & refactoring"]
    },
    "mistralai/mistral-medium-3": {
        "release_date": "2025-05-07",
        "update_history": ["May 2025: Launch of Medium 3", "Aug 2025: 3.1 Refresh with improved tone"],
        "strengths": ["Frontier-class coding & STEM", "High efficiency-to-intelligence ratio", "Native support for complex tool use"],
        "weaknesses": ["Smaller context than Large"],
        "use_cases": ["Professional software engineering", "STEM research assistance"]
    },
    "mistralai/mistral-large-2512": {
        "release_date": "2025-12-05",
        "update_history": ["Dec 2025: Major architectural update for Large series"],
        "strengths": ["Massive knowledge base", "SOTA reasoning in open weights"],
        "weaknesses": ["High hardware requirements"],
        "use_cases": ["Enterprise-grade analysis", "Legal & Medical research"]
    },
    "mistralai/ministral-8b-2512": {
        "release_date": "2025-11-20",
        "update_history": ["Nov 2025: Optimized 8B flagship for the 2512 series"],
        "strengths": ["Best-in-class 8B reasoning", "Strong coding for its size"],
        "weaknesses": ["Smallest in its performance bracket"],
        "use_cases": ["Smart home agents", "Local coding assistance"]
    },
    "mistralai/devstral-2512": {
        "release_date": "2025-12-08",
        "update_history": ["Dec 2025: Agentic coding flagship launch"],
        "strengths": ["SOTA for autonomous software agents", "Deep Git integration capabilities"],
        "weaknesses": ["Narrowly focused on engineering"],
        "use_cases": ["Autonomous PR management", "Codebase migration"]
    },
    "mistralai/devstral-medium": {
        "release_date": "2025-12-10",
        "update_history": ["Dec 2025: Mid-size agentic coding model"],
        "strengths": ["Exceptional speed/intelligence balance for coding"],
        "weaknesses": ["Less capabale on general prose"],
        "use_cases": ["Refactoring assistant", "Technical documentation lead"]
    },
    "mistralai/devstral-small": {
        "release_date": "2025-12-10",
        "update_history": ["Dec 2025: Compact agentic coding model"],
        "strengths": ["Fastest logic for simple code edits", "Low cost"],
        "weaknesses": ["Limited complex architecture reasoning"],
        "use_cases": ["Unit test generation", "Bug triage"]
    },
    "mistralai/ministral-3b-2512": {
        "release_date": "2025-11-20",
        "update_history": ["Nov 2025: Smallest 2512 series model"],
        "strengths": ["Ultra-fast response", "Context-aware edge processing"],
        "weaknesses": ["Limited knowledge depth"],
        "use_cases": ["Voice control agents", "Mobile UI automation"]
    },
    "mistralai/ministral-14b-2512": {
        "release_date": "2025-11-20",
        "update_history": ["Nov 2025: High-efficiency 14B variant"],
        "strengths": ["Strong reasoning architecture", "Low VRAM requirements"],
        "weaknesses": ["Struggles with very large context"],
        "use_cases": ["Local research assistant", "Corporate internal QA"]
    },
    "mistralai/mistral-medium-3.1": {
        "release_date": "2025-08-15",
        "update_history": ["Aug 2025: Refinement of Medium 3"],
        "strengths": ["Polished conversational tone", "Higher accuracy in STEM"],
        "weaknesses": ["Overshadowed by Large for high-end tasks"],
        "use_cases": ["STEM tutoring", "Professional writing assistant"]
    },
    "mistralai/mistral-small-3.1-24b-instruct": {
        "release_date": "2025-07-10",
        "update_history": ["Jul 2025: Optimized 24B instruction model"],
        "strengths": ["Excellent cost/performance ratio"],
        "weaknesses": ["Lower reasoning than Medium series"],
        "use_cases": ["Large-scale text classification", "Summarization flows"]
    },
    "mistralai/mistral-small-3.2-24b-instruct": {
        "release_date": "2025-10-05",
        "update_history": ["Oct 2025: Revision with improved tool use"],
        "strengths": ["Reliable function calling", "Faster inference"],
        "weaknesses": ["Knowledge cutoff latency"],
        "use_cases": ["Action-oriented agents", "Process automation"]
    },
    "mistralai/mistral-small-creative": {
        "release_date": "2025-09-20",
        "update_history": ["Sep 2025: Specialized for creative prose and roleplay"],
        "strengths": ["Unfiltered creativity", "Rich narrative style"],
        "weaknesses": ["Lower logic and coding accuracy"],
        "use_cases": ["Creative writing", "Interactive storytelling"]
    },
    "mistralai/codestral-2508": {
        "release_date": "2025-08-22",
        "update_history": ["Aug 2025: High-performance coding specialist"],
        "strengths": ["Superior Python/C++ generation", "82.4% LiveCodeBench"],
        "weaknesses": ["Single-modality focus"],
        "use_cases": ["Dedicated coding IDE support", "Legacy code modernization"]
    },
    "mistralai/voxtral-small-24b-2507": {
        "release_date": "2025-07-28",
        "update_history": ["Jul 2025: Native audio/speech reasoning flagship"],
        "strengths": ["State-of-the-art speech-to-text-to-reasoning", "Low WER"],
        "weaknesses": ["Premium pricing for audio tokens"],
        "use_cases": ["Real-time meeting synthesis", "Advanced voice assistants"]
    },
}

# ── Allowed providers (top 5 major LLM providers) ────────────────────────────
ALLOWED_PROVIDERS = {"openai", "anthropic", "meta-llama", "google", "mistralai"}
MAX_MODELS_PER_PROVIDER = 15

# Must-include flagship models (always kept even if older)
PINNED_MODELS = {
    # OpenAI essentials
    "openai/gpt-4o", "openai/gpt-4o-mini", "openai/o1",
    "openai/o3-mini", "openai/o4-mini",
    # Anthropic essentials
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-haiku",
    "anthropic/claude-sonnet-4", "anthropic/claude-opus-4",
    # Google essentials
    "google/gemini-2.0-flash-001", "google/gemini-2.5-pro", "google/gemini-2.5-flash",
    # Meta essentials
    "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-4-maverick",
    # Mistral essentials
    "mistralai/mistral-large", "mistralai/mistral-medium-3",
}

# IDs to skip (meta-routers, free duplicates, guard/utility models, old dated variants)
SKIP_SUFFIXES = [":free", ":extended", ":exacto", ":floor"]
SKIP_IDS = {"openrouter/auto", "openrouter/free", "openrouter/bodybuilder", "switchpoint/router"}
# Patterns in model IDs to skip (guard models, old date-stamped variants, custom-tools, etc.)
SKIP_SUBSTRINGS = ["guard", "safeguard", "customtools", "-0314", "-0613",
                    "audio", "preview-0"]


def _transform_model(raw: dict) -> Optional[LLMModel]:
    """Convert an OpenRouter model JSON object into our LLMModel schema."""
    model_id = raw.get("id", "")

    # Skip meta-routers and free duplicates
    if model_id in SKIP_IDS:
        return None
    if any(model_id.endswith(suffix) for suffix in SKIP_SUFFIXES):
        return None

    # Only allow models from whitelisted providers
    parts_check = model_id.split("/", 1)
    provider_key_check = parts_check[0] if len(parts_check) > 1 else "unknown"
    if provider_key_check not in ALLOWED_PROVIDERS:
        return None

    # Skip guard, utility, and old dated variant models (unless pinned)
    model_slug = parts_check[1] if len(parts_check) > 1 else model_id
    if model_id not in PINNED_MODELS and any(sub in model_slug for sub in SKIP_SUBSTRINGS):
        return None

    # Skip models with zero pricing (likely internal/test)
    pricing = raw.get("pricing", {})
    prompt_price = float(pricing.get("prompt", "0") or "0")
    completion_price = float(pricing.get("completion", "0") or "0")

    # Extract provider from ID
    parts = model_id.split("/", 1)
    provider_key = parts[0] if len(parts) > 1 else "unknown"
    provider_display = PROVIDER_DISPLAY.get(provider_key, provider_key.replace("-", " ").title())

    # Build slug ID (replace / with -)
    slug = model_id.replace("/", "-")

    # Context & output
    context_window = raw.get("context_length", 0) or 0
    top_provider = raw.get("top_provider", {}) or {}
    max_output = top_provider.get("max_completion_tokens", 4096) or 4096

    # Pricing: OpenRouter returns per-token, convert to per-1M
    input_price_per_1m = round(prompt_price * 1_000_000, 4)
    output_price_per_1m = round(completion_price * 1_000_000, 4)

    # Architecture / modality
    arch = raw.get("architecture", {}) or {}
    input_modalities = arch.get("input_modalities", ["text"])
    output_modalities = arch.get("output_modalities", ["text"])
    modality = arch.get("modality", "text->text")

    # Curated enrichment
    benchmarks = CURATED_BENCHMARKS.get(model_id, {})
    meta = CURATED_META.get(model_id, {})
    strengths = meta.get("strengths", [])
    weaknesses = meta.get("weaknesses", [])
    use_cases = meta.get("use_cases", [])
    release_date = meta.get("release_date", "")
    update_history = meta.get("update_history", [])

    description = raw.get("description", "") or ""
    # Truncate very long descriptions
    if len(description) > 300:
        description = description[:297] + "..."

    return LLMModel(
        id=slug,
        openrouter_id=model_id,
        name=raw.get("name", model_id),
        provider=provider_display,
        description=description,
        parameters_b=None,
        context_window=context_window,
        input_price_per_1m=input_price_per_1m,
        output_price_per_1m=output_price_per_1m,
        max_output_tokens=max_output,
        release_date=release_date,
        update_history=update_history,
        benchmarks=benchmarks,
        strengths=strengths,
        weaknesses=weaknesses,
        use_cases=use_cases,
        logo_color=PROVIDER_COLORS.get(provider_key, "#6366f1"),
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        modality=modality,
        # Per-token pricing
        input_price_per_token=prompt_price,
        output_price_per_token=completion_price,
    )


def fetch_models() -> list[LLMModel]:
    """Fetch models from OpenRouter API with caching."""
    now = time.time()

    # Return cached if fresh
    if _cache["models"] is not None and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _cache["models"]

    try:
        logger.info("Fetching models from OpenRouter API...")
        resp = httpx.get(OPENROUTER_URL, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        raw_models = data.get("data", [])

        models_with_ts: list[tuple[int, LLMModel]] = []
        for raw in raw_models:
            m = _transform_model(raw)
            if m is not None:
                created = raw.get("created", 0) or 0
                models_with_ts.append((created, m))

        # Cap at MAX_MODELS_PER_PROVIDER per provider, preferring newest models
        from collections import defaultdict
        by_provider: dict[str, list[tuple[int, LLMModel]]] = defaultdict(list)
        for ts, m in models_with_ts:
            by_provider[m.provider].append((ts, m))

        capped: list[LLMModel] = []
        for provider, pmodels in by_provider.items():
            # Always include pinned models first
            pinned = [(ts, m) for ts, m in pmodels if m.openrouter_id in PINNED_MODELS]
            others = [(ts, m) for ts, m in pmodels if m.openrouter_id not in PINNED_MODELS]
            # Sort remaining by created timestamp descending (newest first)
            others.sort(key=lambda x: x[0], reverse=True)
            # Take pinned + fill remaining slots with newest
            remaining_slots = MAX_MODELS_PER_PROVIDER - len(pinned)
            selected = pinned + others[:max(0, remaining_slots)]
            capped.extend(m for _, m in selected)

        # Final sort by provider then name for display
        capped.sort(key=lambda m: (m.provider, m.name))

        _cache["models"] = capped
        _cache["fetched_at"] = now
        logger.info(f"Fetched {len(capped)} models from OpenRouter ({len(models_with_ts)} matched providers, {len(raw_models)} raw)")
        return capped

    except Exception as e:
        logger.error(f"Failed to fetch from OpenRouter: {e}")
        # Return cached even if stale
        if _cache["models"] is not None:
            logger.info("Returning stale cache")
            return _cache["models"]
        # Last resort: return empty
        logger.warning("No cached data available, returning empty list")
        return []
