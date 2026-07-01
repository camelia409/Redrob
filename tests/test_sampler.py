"""Tests for the stratified hand-labeling sampler."""
from src.evaluation.stratified_sampler import sample_30_for_labeling


def test_sampler_deterministic():
    ids_a = sample_30_for_labeling(seed=42)
    ids_b = sample_30_for_labeling(seed=42)
    assert ids_a == ids_b


def test_sampler_returns_30_ids():
    ids = sample_30_for_labeling(seed=42)
    assert len(ids) == 30


def test_sampler_no_overlapping_strata():
    ids = sample_30_for_labeling(seed=42)
    assert len(set(ids)) == 30
