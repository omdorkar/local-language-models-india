"""Single source of truth for languages, datasets and candidate models.

Every dataset here is public and was produced by its original authors; this project
collects no new human data. Licences are recorded so derived releases can honour them.
"""

from dataclasses import dataclass, field

SEED = 1909  # fixed seed; used for sampling, decoding and bootstrap


@dataclass(frozen=True)
class Language:
    name: str
    iso: str  # ISO 639-1 where it exists
    belebele_native: str | None = None  # Belebele config, native script
    belebele_latn: str | None = None  # Belebele config, romanized (human-transliterated)
    global_mmlu: str | None = None  # Global-MMLU config
    global_mmlu_lite: str | None = None  # Global-MMLU-Lite config (fully human-translated)
    milu: str | None = None  # MILU config
    dakshina: str | None = None  # Dakshina language folder


# Languages of India (8th Schedule of the Constitution) that the open benchmarks cover.
# Coverage was checked against each dataset's Hugging Face file listing on 2026-10-20.
LANGUAGES: dict[str, Language] = {
    "en": Language("English", "en", belebele_native="eng_Latn", global_mmlu="en", global_mmlu_lite="en", milu="English"),
    "hi": Language("Hindi", "hi", "hin_Deva", "hin_Latn", "hi", "hi", "Hindi", "hi"),
    "bn": Language("Bengali", "bn", "ben_Beng", "ben_Latn", "bn", "bn", "Bengali", "bn"),
    "ur": Language("Urdu", "ur", "urd_Arab", "urd_Latn", dakshina="ur"),
    "ne": Language("Nepali", "ne", "npi_Deva", "npi_Latn", "ne"),
    "mr": Language("Marathi", "mr", "mar_Deva", milu="Marathi", dakshina="mr"),
    "te": Language("Telugu", "te", "tel_Telu", global_mmlu="te", milu="Telugu", dakshina="te"),
    "ta": Language("Tamil", "ta", "tam_Taml", milu="Tamil", dakshina="ta"),
    "gu": Language("Gujarati", "gu", "guj_Gujr", milu="Gujarati", dakshina="gu"),
    "kn": Language("Kannada", "kn", "kan_Knda", milu="Kannada", dakshina="kn"),
    "ml": Language("Malayalam", "ml", "mal_Mlym", milu="Malayalam", dakshina="ml"),
    "pa": Language("Punjabi", "pa", "pan_Guru", milu="Punjabi", dakshina="pa"),
    "or": Language("Odia", "or", "ory_Orya", global_mmlu_lite="or", milu="Odia"),
    "as": Language("Assamese", "as", "asm_Beng"),
}


@dataclass(frozen=True)
class Dataset:
    key: str
    repo: str  # Hugging Face dataset repo, or a URL for non-HF sources
    licence: str
    citation: str
    gated: bool = False
    patterns: list[str] = field(default_factory=list)


def _belebele_patterns() -> list[str]:
    configs = [c for l in LANGUAGES.values() for c in (l.belebele_native, l.belebele_latn) if c]
    return [f"data/{c}.jsonl" for c in configs] + ["README.md"]


DATASETS: dict[str, Dataset] = {
    "belebele": Dataset(
        "belebele", "facebook/belebele", "CC-BY-SA-4.0",
        "Bandarkar et al., The Belebele Benchmark, ACL 2024",
        patterns=_belebele_patterns(),
    ),
    "global_mmlu": Dataset(
        "global_mmlu", "CohereLabs/Global-MMLU", "Apache-2.0",
        "Singh et al., Global MMLU, ACL 2025",
        patterns=[f"{l.global_mmlu}/*.parquet" for l in LANGUAGES.values() if l.global_mmlu] + ["README.md"],
    ),
    "global_mmlu_lite": Dataset(
        "global_mmlu_lite", "CohereLabs/Global-MMLU-Lite", "Apache-2.0",
        "Singh et al., Global MMLU, ACL 2025 (Lite: human-translated subset)",
        patterns=[f"{l.global_mmlu_lite}/*" for l in LANGUAGES.values() if l.global_mmlu_lite] + ["README.md"],
    ),
    "milu": Dataset(
        "milu", "ai4bharat/MILU", "CC-BY-4.0",
        "MILU: A Multi-task Indic Language Understanding Benchmark, arXiv:2411.02538",
        gated=True,
        patterns=[f"{l.milu}/*.parquet" for l in LANGUAGES.values() if l.milu] + ["README.md"],
    ),
    "comi_lingua": Dataset(
        "comi_lingua", "LingoIITGN/COMI-LINGUA", "CC-BY-4.0",
        "COMI-LINGUA: Expert Annotated Dataset for Hindi-English Code-Mixing, arXiv:2503.21670",
        # Test splits only: enough real Hinglish to validate generated code-mixing.
        patterns=["MT_test.csv", "LID_test.csv", "MLI_test.csv", "README.md"],
    ),
    "dakshina": Dataset(
        "dakshina", "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar",
        "CC-BY-SA-4.0",
        "Roark et al., Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset, LREC 2020",
    ),
}


@dataclass(frozen=True)
class ModelCandidate:
    key: str
    display: str
    ollama_tag: str | None  # tag for `ollama pull`, if Ollama supports it
    gguf_repo: str | None  # GGUF repo for llama.cpp, if Ollama does not
    origin: str
    note: str = ""


# Exact digests: reports/model_versions.md (python -m evaluation.versions).
MODELS: list[ModelCandidate] = [
    ModelCandidate("gemma3-12b", "Gemma 3 12B", "gemma3:12b", None, "Google (US)"),
    ModelCandidate("qwen3-8b", "Qwen3 8B", "qwen3:8b", None, "Alibaba (China)", "reasoning_effort none"),
    ModelCandidate("llama3.1-8b", "Llama 3.1 8B", "llama3.1:8b", None, "Meta (US)", "comparable to MIT 2026 study"),
    ModelCandidate("gpt-oss-20b", "gpt-oss 20B", "gpt-oss:20b", None, "OpenAI (US)", "reasoning_effort low, max_tokens 512; reasoning cannot be disabled"),
    ModelCandidate(
        "sarvam-30b", "Sarvam-30B", None, "Sumitc13/sarvam-30b-GGUF", "Sarvam AI (India)",
        "MoE, 2.4B active; BF16 quantized locally to Q3_K_M; llama-server --reasoning-budget 0 (reports/sarvam_setup.md)",
    ),
]
