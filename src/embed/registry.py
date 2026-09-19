"""Model ID -> loader mapping for every embedding model in the lineup.

Responsibility: given a key from config.Config.embedding_models, return a
loaded encoder object with a uniform .encode(texts) -> np.ndarray interface,
handling the different loading paths required by sentence-transformers models
vs. mean-pooled MLM models (e.g. HingRoBERTa). Built in Phase 4.

All six lineup models -- including l3cube-pune/hing-roberta, a plain MLM
checkpoint with no native sentence-embedding head -- load cleanly through
`sentence_transformers.SentenceTransformer(model_id)` in this environment
(confirmed in Phase 0, results/logs/phase0_verify_env.json): when a checkpoint
has no ST `modules.json`, the library auto-wraps it as Transformer + mean
Pooling, which is exactly the naive mean-pooled behaviour PROJECT.md calls for
("MLM not a sentence encoder -- this is the point"). So one loader path
covers the whole lineup; no separate transformers/AutoModel branch is needed.

multilingual-e5-large is the one model in the lineup whose own model card
requires an instruction prefix even for symmetric similarity ("query: " on
both sides for non-retrieval tasks) -- encoded here as `query_prefix` so every
caller gets it automatically rather than having to remember it per call site.
"""

from __future__ import annotations

import numpy as np

from config import Config

# model_key -> prefix prepended to every text before encoding. Only
# multilingual-e5-large needs this (see module docstring); every other model
# in the lineup is used with its raw text unprefixed.
_QUERY_PREFIXES: dict[str, str] = {
    "multilingual_e5": "query: ",
}


class Encoder:
    """Uniform wrapper around a loaded SentenceTransformer model.

    .encode(texts) always returns an (len(texts), embedding_dim) float32
    array, L2-normalized so cosine similarity reduces to a dot product
    (src/cache/store.py relies on this).
    """

    def __init__(self, key: str, model_id: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.key = key
        self.model_id = model_id
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(model_id, trust_remote_code=True)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        inputs = [self.query_prefix + t for t in texts] if self.query_prefix else list(texts)
        vectors = self._model.encode(
            inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def __repr__(self) -> str:
        return f"Encoder(key={self.key!r}, model_id={self.model_id!r}, dim={self.embedding_dim})"


# In-process cache: loading a model (disk read + weight init) is expensive
# enough (seconds, per phase0_verify_env.json) that repeated get_encoder()
# calls in the same run must not reload it.
_loaded: dict[str, Encoder] = {}


def get_encoder(model_key: str, config: Config | None = None) -> Encoder:
    """Return a loaded Encoder for `model_key`, a key in
    Config.embedding_models (e.g. "english_centric", "hingroberta"). Loaded
    models are cached in-process so repeated calls are free after the first.
    """
    config = config or Config()
    if model_key not in config.embedding_models:
        raise KeyError(
            f"Unknown model_key {model_key!r}. Valid keys: {sorted(config.embedding_models)}"
        )
    if model_key not in _loaded:
        model_id = config.embedding_models[model_key]
        prefix = _QUERY_PREFIXES.get(model_key, "")
        _loaded[model_key] = Encoder(model_key, model_id, query_prefix=prefix)
    return _loaded[model_key]


def clear_loaded_cache() -> None:
    """Drop all in-process loaded models (frees memory; mainly for tests)."""
    _loaded.clear()
