"""Bridge: preprocessing fixes for Indian-language LLM evaluation (romanized -> native script; native -> English)."""
from .pipeline import normalize, to_english

__all__ = ["normalize", "to_english"]
