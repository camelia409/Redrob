"""Tests for the grounded, rank-aware reasoning generator."""
import pytest

from src.ingestion.loader import load_candidates_sample
from src.reasoning.generator import generate
from src.reasoning.grounding import HallucinationError, assert_grounded


def _candidate(idx: int = 0):
    return load_candidates_sample(10)[idx]


def _empty_features():
    return {}


def test_generator_is_deterministic():
    c = _candidate(0)
    text1 = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    text2 = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    assert text1 == text2


def test_reasoning_length_within_bounds():
    for i in range(10):
        c = _candidate(i)
        for rank in (3, 15, 45, 85):
            text = generate(c, rank=rank, score=1.0, features_row=_empty_features(), honeypot_flags=[])
            assert 60 <= len(text) <= 200, f"rank={rank} len={len(text)} text={text!r}"
            assert_grounded(c, text)


def test_top10_lead_is_enthusiastic():
    c = _candidate(0)
    text = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    assert any(word in text.lower() for word in ("strong", "standout", "top-fit", "senior", "clear"))
    assert "weak" not in text.lower() and "concerning" not in text.lower()


def test_rank_11_30_lead_is_confident():
    c = _candidate(1)
    text = generate(c, rank=25, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    # Should still be positive; no concern-forward language.
    assert "weak" not in text.lower()
    assert "concerning" not in text.lower()


def test_rank_31_70_includes_measured_language():
    c = _candidate(2)
    text = generate(c, rank=55, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    # A concern clause is expected for this band.
    assert any(word in text.lower() for word in ("concern", "gap", "flagged", "caution", "issue", "caveat"))


def test_rank_71_100_is_concern_forward():
    c = _candidate(3)
    text = generate(c, rank=85, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    assert any(word in text.lower() for word in ("weak", "concerning", "low-confidence", "poor", "risky"))
    assert any(word in text.lower() for word in ("concern", "gap", "flagged", "caution", "issue"))


def test_rank_bands_produce_different_texts():
    c = _candidate(4)
    t_top = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    t_mid = generate(c, rank=25, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    t_low = generate(c, rank=85, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    assert len({t_top, t_mid, t_low}) == 3


def test_grounding_catches_fake_skill():
    c = _candidate(0)
    text = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    hallucinated = text + " and tensorflow"
    with pytest.raises(HallucinationError):
        assert_grounded(c, hallucinated)


def test_grounding_catches_fake_company():
    c = _candidate(0)
    text = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    hallucinated = text + " at openai"
    with pytest.raises(HallucinationError):
        assert_grounded(c, hallucinated)


def test_grounding_catches_fake_city():
    c = _candidate(0)
    text = generate(c, rank=5, score=1.0, features_row=_empty_features(), honeypot_flags=[])
    hallucinated = text + " in bangalore"
    with pytest.raises(HallucinationError):
        assert_grounded(c, hallucinated)


def test_opening_phrase_variation_across_candidates():
    candidates = load_candidates_sample(10)
    openings = set()
    for i, c in enumerate(candidates):
        rank = i + 1
        text = generate(c, rank=rank, score=1.0, features_row=_empty_features(), honeypot_flags=[])
        assert_grounded(c, text)
        # First 3 words serve as a proxy for the lead phrase.
        openings.add(" ".join(text.split()[:3]).lower())
    assert len(openings) >= 7, f"Only {len(openings)} distinct openings: {openings}"
