"""Unit tests for src/embed/registry.py that don't require loading a real
model (no network, no multi-second model load). Registry's actual loading
path is exercised for real in scripts/phase4_sanity_check.py.
"""

import pytest

from config import Config
from embed.registry import get_encoder


def test_unknown_model_key_raises_before_loading_anything():
    with pytest.raises(KeyError, match="Unknown model_key"):
        get_encoder("not_a_real_model_key", Config())


def test_error_message_lists_valid_keys():
    with pytest.raises(KeyError) as exc_info:
        get_encoder("bogus", Config())
    message = str(exc_info.value)
    for key in Config().embedding_models:
        assert key in message
