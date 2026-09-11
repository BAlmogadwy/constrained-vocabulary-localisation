from .gpt5_detector import GPT54MiniDetector, GPT55Detector, GPT5Detector  # noqa: F401
from .claude_detector import ClaudeHaiku45Detector, ClaudeHaikuDetector, ClaudeOpus48Detector  # noqa: F401
from .gemini_detector import Gemini31ProDetector, Gemini35FlashDetector, GeminiFlashDetector  # noqa: F401
from .gemma_detector import GemmaDetector  # noqa: F401
from .mistral_detector import MistralDetector  # noqa: F401
from .openai_compatible_detector import (  # noqa: F401
    DashScopeQwenVLMaxDetector,
    MistralPixtralLargeDetector,
    OpenRouterGemma327BDetector,
    OpenRouterLlama32Vision11BDetector,
    OpenRouterLlama4MaverickDetector,
    OpenRouterMistralLargeDetector,
    OpenRouterPixtralLargeDetector,
    OpenRouterQwen25VL72BDetector,
    OpenRouterQwen3VL235BDetector,
)
from .qwen_detector import QwenDetector  # noqa: F401


def _optional_local_imports():
    try:
        from .llava_detector import LlavaDetector, LlavaMistralDetector  # noqa: F401
        from .llama_vision_detector import LlamaVisionDetector  # noqa: F401
        from .gemma_local_detector import GemmaLocalDetector  # noqa: F401
        from .paligemma_local_detector import PaliGemmaLocalDetector  # noqa: F401
        from .qwen3_local_detector import Qwen3LocalDetector  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "transformers"}:
            raise
        return {
            "LlavaDetector": None,
            "LlavaMistralDetector": None,
            "LlamaVisionDetector": None,
            "GemmaLocalDetector": None,
            "PaliGemmaLocalDetector": None,
            "Qwen3LocalDetector": None,
        }
    return {
        "LlavaDetector": LlavaDetector,
        "LlavaMistralDetector": LlavaMistralDetector,
        "LlamaVisionDetector": LlamaVisionDetector,
        "GemmaLocalDetector": GemmaLocalDetector,
        "PaliGemmaLocalDetector": PaliGemmaLocalDetector,
        "Qwen3LocalDetector": Qwen3LocalDetector,
    }


_LOCAL = _optional_local_imports()
LlavaDetector = _LOCAL["LlavaDetector"]
LlavaMistralDetector = _LOCAL["LlavaMistralDetector"]
LlamaVisionDetector = _LOCAL["LlamaVisionDetector"]
GemmaLocalDetector = _LOCAL["GemmaLocalDetector"]
PaliGemmaLocalDetector = _LOCAL["PaliGemmaLocalDetector"]
Qwen3LocalDetector = _LOCAL["Qwen3LocalDetector"]


MODEL_REGISTRY = {
    "gpt-5": GPT5Detector,
    "gpt-5.5": GPT55Detector,
    "gpt-5.4-mini": GPT54MiniDetector,
    "claude-haiku-4.5": ClaudeHaiku45Detector,
    "claude-haiku-4-5": ClaudeHaiku45Detector,
    "claude-opus-4.8": ClaudeOpus48Detector,
    "claude-opus-4-8": ClaudeOpus48Detector,
    "gemini-2.5-flash": GeminiFlashDetector,
    "gemini-3.1-pro": Gemini31ProDetector,
    "gemini-3.5-flash": Gemini35FlashDetector,
    "gemma-3": GemmaDetector,
    "qwen2.5-vl-72b": QwenDetector,
    "qwen-vl-max": DashScopeQwenVLMaxDetector,
    "qwen2.5-vl-72b-openrouter": OpenRouterQwen25VL72BDetector,
    "qwen3-vl-235b-openrouter": OpenRouterQwen3VL235BDetector,
    "llama-4-maverick-openrouter": OpenRouterLlama4MaverickDetector,
    "llama-3.2-11b-vision-openrouter": OpenRouterLlama32Vision11BDetector,
    "gemma-3-27b-openrouter": OpenRouterGemma327BDetector,
    "mistral-large-openrouter": OpenRouterMistralLargeDetector,
    "pixtral-large-openrouter": OpenRouterPixtralLargeDetector,
    "pixtral-large-mistral": MistralPixtralLargeDetector,
    "mistral-pixtral-large": MistralDetector,
}

if LlavaDetector is not None:
    MODEL_REGISTRY.update(
        {
            "llava-1.6": LlavaDetector,
            "llava-1.5-7b-hf": LlavaDetector,
        }
    )
if LlavaMistralDetector is not None:
    MODEL_REGISTRY["llava-v1.6-mistral-7b-hf"] = LlavaMistralDetector
if LlamaVisionDetector is not None:
    MODEL_REGISTRY.update(
        {
            "llama-3.2-vision-11b": LlamaVisionDetector,
            "Llama-3.2-11B-Vision-Instruct": LlamaVisionDetector,
            "meta-llama/Llama-3.2-11B-Vision-Instruct": LlamaVisionDetector,
        }
    )
if GemmaLocalDetector is not None:
    MODEL_REGISTRY.update(
        {
            "gemma-3-12b-it": GemmaLocalDetector,
            "google/gemma-3-12b-it": GemmaLocalDetector,
        }
    )
if PaliGemmaLocalDetector is not None:
    MODEL_REGISTRY.update(
        {
            "paligemma2-10b-mix-448": PaliGemmaLocalDetector,
            "google/paligemma2-10b-mix-448": PaliGemmaLocalDetector,
        }
    )
if Qwen3LocalDetector is not None:
    MODEL_REGISTRY["Qwen3-VL-8B-Instruct"] = Qwen3LocalDetector


MODEL_API_ENV = {
    "gpt-5": "OPENAI_API_KEY",
    "gpt-5.5": "OPENAI_API_KEY",
    "gpt-5.4-mini": "OPENAI_API_KEY",
    "claude-haiku-4.5": "ANTHROPIC_API_KEY",
    "claude-haiku-4-5": "ANTHROPIC_API_KEY",
    "claude-opus-4.8": "ANTHROPIC_API_KEY",
    "claude-opus-4-8": "ANTHROPIC_API_KEY",
    "gemini-2.5-flash": "GOOGLE_API_KEY",
    "gemini-3.1-pro": "GOOGLE_API_KEY",
    "gemini-3.5-flash": "GOOGLE_API_KEY",
    "gemma-3": "GOOGLE_API_KEY",
    "qwen2.5-vl-72b": "QWEN_API_KEY",
    "qwen-vl-max": "QWEN_API_KEY",
    "qwen2.5-vl-72b-openrouter": "OPENROUTER_API_KEY",
    "qwen3-vl-235b-openrouter": "OPENROUTER_API_KEY",
    "llama-4-maverick-openrouter": "OPENROUTER_API_KEY",
    "llama-3.2-11b-vision-openrouter": "OPENROUTER_API_KEY",
    "gemma-3-27b-openrouter": "OPENROUTER_API_KEY",
    "mistral-large-openrouter": "OPENROUTER_API_KEY",
    "pixtral-large-openrouter": "OPENROUTER_API_KEY",
    "pixtral-large-mistral": "MISTRAL_API_KEY",
    "mistral-pixtral-large": "MISTRAL_API_KEY",
}

for local_name in (
    "llava-1.6",
    "llava-1.5-7b-hf",
    "llava-v1.6-mistral-7b-hf",
    "llama-3.2-vision-11b",
    "Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "gemma-3-12b-it",
    "google/gemma-3-12b-it",
    "paligemma2-10b-mix-448",
    "google/paligemma2-10b-mix-448",
    "Qwen3-VL-8B-Instruct",
):
    if local_name in MODEL_REGISTRY:
        MODEL_API_ENV[local_name] = None
