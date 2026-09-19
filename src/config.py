"""Project-wide configuration: paths, model registry, sweep range, seed, FHR budget.

No logic lives here. Every other module reads from this dataclass instead of
hardcoding paths or model IDs.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Repo root = parent of the `src/` directory this file lives in.
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- Paths -----------------------------------------------------------
    repo_root: Path = _REPO_ROOT
    notes_dir: Path = _REPO_ROOT / "notes"
    data_dir: Path = _REPO_ROOT / "data"
    data_raw_dir: Path = _REPO_ROOT / "data" / "raw"
    data_annotated_dir: Path = _REPO_ROOT / "data" / "annotated"
    data_release_dir: Path = _REPO_ROOT / "data" / "release"
    guidelines_dir: Path = _REPO_ROOT / "data" / "guidelines"
    results_dir: Path = _REPO_ROOT / "results"
    results_raw_dir: Path = _REPO_ROOT / "results" / "raw"
    results_tables_dir: Path = _REPO_ROOT / "results" / "tables"
    results_figures_dir: Path = _REPO_ROOT / "results" / "figures"
    results_logs_dir: Path = _REPO_ROOT / "results" / "logs"
    paper_dir: Path = _REPO_ROOT / "paper"
    # On-disk vector cache: results/embeddings/<model_key>/<text_hash>.npy.
    # Binary + regenerable, so *.npy is already gitignored globally; this
    # just names where embed/encode.py keeps them. Built in Phase 4.
    embed_cache_dir: Path = _REPO_ROOT / "results" / "embeddings"
    # On-disk word-transliteration cache: results/translit_cache/<word_hash>.json,
    # {"word": ..., "result": ...}. Keeps re-runs of exp04 (and any other
    # caller) from re-paying a WSL round trip for a word already seen.
    # Built in Phase 6.
    translit_cache_dir: Path = _REPO_ROOT / "results" / "translit_cache"

    # --- Embedding model registry: role -> HF model ID -------------------
    embedding_models: dict = field(default_factory=lambda: {
        "english_centric": "sentence-transformers/all-MiniLM-L6-v2",
        "multilingual_e5": "intfloat/multilingual-e5-large",
        "bge_m3": "BAAI/bge-m3",
        "labse": "sentence-transformers/LaBSE",
        "indic_sbert": "l3cube-pune/indic-sentence-similarity-sbert",
        "hingroberta": "l3cube-pune/hing-roberta",
        # Added 2026-09-19 for the blog follow-up, not part of the submitted
        # paper's lineup. Released 2026-05-14; no query prefix per its card.
        "granite_r2": "ibm-granite/granite-embedding-97m-multilingual-r2",
    })

    # --- Cache decision threshold sweep -----------------------------------
    threshold_min: float = 0.60
    threshold_max: float = 0.98
    threshold_step: float = 0.01

    # --- Reproducibility ---------------------------------------------------
    random_seed: int = 42

    # --- Target operating point ---------------------------------------------
    target_false_hit_rate_budget: float = 0.02
