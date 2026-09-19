"""Unit tests for src/dataset/build.py's three-way join, against small
hand-written JSONL fixtures (not the real 1162-record dataset)."""

import json

import pytest

from config import Config
from dataset.build import load_three_way


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _prov():
    return {"source": "manual", "author": "test", "created_at": "2026-08-14",
            "license_note": "", "needs_review": False}


def _record(intent_id, role, condition, seq, text, equivalence_class_id=None):
    return {
        "record_id": f"{intent_id}__{role}__{condition}__{seq:03d}",
        "intent_id": intent_id,
        "equivalence_class_id": equivalence_class_id or intent_id,
        "condition": condition,
        "role": role,
        "text": text,
        "provenance": _prov(),
    }


def _cfg(tmp_path):
    d = tmp_path / "annotated"
    d.mkdir()
    return Config(data_annotated_dir=d), d


def test_load_three_way_joins_matching_records(tmp_path):
    cfg, d = _cfg(tmp_path)
    en = [_record("account.reset_password", "canonical", "en", 1, "How do I reset my password?")]
    hi_deva = [_record("account.reset_password", "canonical", "hi_deva", 1, "पासवर्ड रीसेट")]
    hi_rom = [_record("account.reset_password", "canonical", "hi_rom", 1, "password reset kaise kare")]
    _write_jsonl(d / "en.jsonl", en)
    _write_jsonl(d / "hi_deva.jsonl", hi_deva)
    _write_jsonl(d / "hi_rom.jsonl", hi_rom)

    triples = load_three_way(cfg)
    assert len(triples) == 1
    t = triples[0]
    assert t.intent_id == "account.reset_password"
    assert t.role == "canonical"
    assert t.en.text == "How do I reset my password?"
    assert t.hi_rom.text == "password reset kaise kare"


def test_load_three_way_missing_counterpart_raises(tmp_path):
    cfg, d = _cfg(tmp_path)
    en = [_record("account.reset_password", "canonical", "en", 1, "How do I reset my password?")]
    hi_deva = [_record("account.reset_password", "canonical", "hi_deva", 1, "पासवर्ड रीसेट")]
    hi_rom = []  # missing counterpart
    _write_jsonl(d / "en.jsonl", en)
    _write_jsonl(d / "hi_deva.jsonl", hi_deva)
    _write_jsonl(d / "hi_rom.jsonl", hi_rom)

    with pytest.raises(ValueError, match="not perfectly parallel"):
        load_three_way(cfg)


def test_load_three_way_mismatched_equivalence_class_raises(tmp_path):
    cfg, d = _cfg(tmp_path)
    en = [_record("account.reset_password", "canonical", "en", 1, "text", equivalence_class_id="account.reset_password")]
    hi_deva = [_record("account.reset_password", "canonical", "hi_deva", 1, "text", equivalence_class_id="account.reset_password")]
    # hi_rom accidentally resolves to a different equivalence class
    hi_rom = [_record("account.reset_password", "canonical", "hi_rom", 1, "text", equivalence_class_id="billing.refund_status")]
    _write_jsonl(d / "en.jsonl", en)
    _write_jsonl(d / "hi_deva.jsonl", hi_deva)
    _write_jsonl(d / "hi_rom.jsonl", hi_rom)

    with pytest.raises(ValueError, match="not perfectly parallel"):
        load_three_way(cfg)


def test_load_three_way_multiple_triples_all_matched(tmp_path):
    cfg, d = _cfg(tmp_path)
    intents = ["account.reset_password", "billing.refund_status"]
    en, hi_deva, hi_rom = [], [], []
    for intent in intents:
        en.append(_record(intent, "canonical", "en", 1, f"{intent} en"))
        hi_deva.append(_record(intent, "canonical", "hi_deva", 1, f"{intent} deva"))
        hi_rom.append(_record(intent, "canonical", "hi_rom", 1, f"{intent} rom"))
    _write_jsonl(d / "en.jsonl", en)
    _write_jsonl(d / "hi_deva.jsonl", hi_deva)
    _write_jsonl(d / "hi_rom.jsonl", hi_rom)

    triples = load_three_way(cfg)
    assert len(triples) == 2
    assert {t.intent_id for t in triples} == set(intents)
