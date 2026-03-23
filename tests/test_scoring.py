"""Tests for agretopicity, foreignness, and ranking scorers."""

from __future__ import annotations

import numpy as np
import pytest

from neoantigen_pipeline.config import ScoringConfig
from neoantigen_pipeline.prediction.results import ScoredCandidate
from neoantigen_pipeline.scoring.agretopicity import AgretopicityScorer
from neoantigen_pipeline.scoring.foreignness import ForeignnessScorer
from neoantigen_pipeline.scoring.ranking import RankingScorer


def _make_scored_candidate(
    peptide: str = "SIINFEKLV",
    wildtype_peptide: str = "SIINFEKLA",
    gene: str = "GENE1",
    mutation_str: str = "GENE1_p.Val9Ala",
    best_allele: str = "HLA-A*02:01",
    presentation_score: float = 0.5,
    binding_affinity_nm: float = 200.0,
    wildtype_affinity_nm: float = 500.0,
    processing_score: float = 0.6,
    agretopicity: float = 2.5,
    expression: float = 50.0,
    vaf: float = 0.3,
) -> ScoredCandidate:
    return ScoredCandidate(
        peptide=peptide,
        wildtype_peptide=wildtype_peptide,
        gene=gene,
        mutation_str=mutation_str,
        best_allele=best_allele,
        presentation_score=presentation_score,
        binding_affinity_nm=binding_affinity_nm,
        wildtype_affinity_nm=wildtype_affinity_nm,
        processing_score=processing_score,
        agretopicity=agretopicity,
        expression=expression,
        vaf=vaf,
    )


class TestAgretopicityScorer:
    """Tests for AgretopicityScorer."""

    def setup_method(self):
        self.scorer = AgretopicityScorer()

    def test_basic_computation(self):
        result = self.scorer.compute(100.0, 1000.0)
        assert result == pytest.approx(10.0)

    def test_equal_affinity_returns_one(self):
        result = self.scorer.compute(500.0, 500.0)
        assert result == pytest.approx(1.0)

    def test_mutant_affinity_zero_returns_zero(self):
        result = self.scorer.compute(0.0, 1000.0)
        assert result == 0.0

    def test_wildtype_stronger_returns_less_than_one(self):
        result = self.scorer.compute(500.0, 100.0)
        assert result == pytest.approx(0.2)

    def test_compute_batch_basic(self):
        scores = self.scorer.compute_batch([100.0, 500.0], [1000.0, 1000.0])
        assert scores[0] == pytest.approx(10.0)
        assert scores[1] == pytest.approx(2.0)

    def test_compute_batch_zero_mutant(self):
        scores = self.scorer.compute_batch([0.0], [500.0])
        assert scores[0] == 0.0

    def test_compute_batch_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            self.scorer.compute_batch([100.0, 200.0], [500.0])

    def test_compute_batch_returns_list(self):
        scores = self.scorer.compute_batch([100.0], [200.0])
        assert isinstance(scores, list)


class TestForeignnessScorer:
    """Tests for ForeignnessScorer."""

    def setup_method(self):
        self.scorer = ForeignnessScorer()

    def test_zero_similarity_is_fully_foreign(self):
        assert self.scorer.compute(0.0) == pytest.approx(1.0)

    def test_full_similarity_has_zero_foreignness(self):
        assert self.scorer.compute(1.0) == pytest.approx(0.0)

    def test_intermediate_value(self):
        assert self.scorer.compute(0.4) == pytest.approx(0.6)

    def test_invalid_similarity_raises(self):
        with pytest.raises(ValueError):
            self.scorer.compute(1.5)
        with pytest.raises(ValueError):
            self.scorer.compute(-0.1)

    def test_compute_batch(self):
        results = self.scorer.compute_batch([0.0, 0.5, 1.0])
        assert results == pytest.approx([1.0, 0.5, 0.0])


class TestRankingScorer:
    """Tests for RankingScorer."""

    def _make_config(self, **kwargs) -> ScoringConfig:
        defaults = dict(
            presentation_score_weight=0.4,
            agretopicity_weight=0.2,
            expression_weight=0.2,
            vaf_weight=0.2,
        )
        defaults.update(kwargs)
        return ScoringConfig(**defaults)

    def _make_candidates(self) -> list[ScoredCandidate]:
        return [
            _make_scored_candidate(
                peptide="PEPTIDE1",
                presentation_score=0.9,
                agretopicity=5.0,
                expression=100.0,
                vaf=0.5,
            ),
            _make_scored_candidate(
                peptide="PEPTIDE2",
                presentation_score=0.5,
                agretopicity=2.0,
                expression=50.0,
                vaf=0.3,
            ),
            _make_scored_candidate(
                peptide="PEPTIDE3",
                presentation_score=0.1,
                agretopicity=1.0,
                expression=10.0,
                vaf=0.1,
            ),
        ]

    def test_rank_returns_neoantigen_candidates(self):
        from neoantigen_pipeline.results.neoantigen import NeoantigenCandidate

        scorer = RankingScorer(self._make_config())
        result = scorer.rank(self._make_candidates())
        assert all(isinstance(c, NeoantigenCandidate) for c in result)

    def test_rank_assigns_sequential_ranks(self):
        scorer = RankingScorer(self._make_config())
        result = scorer.rank(self._make_candidates())
        assert [c.composite_rank for c in result] == [1, 2, 3]

    def test_rank_sorted_descending_by_composite_score(self):
        scorer = RankingScorer(self._make_config())
        result = scorer.rank(self._make_candidates())
        scores = [c.composite_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_top_candidate_has_highest_scores(self):
        scorer = RankingScorer(self._make_config())
        result = scorer.rank(self._make_candidates())
        assert result[0].peptide == "PEPTIDE1"

    def test_rank_composite_score_in_zero_one_range(self):
        scorer = RankingScorer(self._make_config())
        result = scorer.rank(self._make_candidates())
        assert all(0.0 <= c.composite_score <= 1.0 for c in result)

    def test_rank_empty_raises(self):
        scorer = RankingScorer(self._make_config())
        with pytest.raises(ValueError):
            scorer.rank([])

    def test_normalise_constant_returns_zeros(self):
        _ = RankingScorer(self._make_config())
        arr = np.array([0.5, 0.5, 0.5])
        result = RankingScorer._normalise(arr)
        assert np.all(result == 0.0)

    def test_normalise_range(self):
        _ = RankingScorer(self._make_config())
        arr = np.array([0.0, 0.5, 1.0])
        result = RankingScorer._normalise(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_rank_preserves_all_candidate_fields(self):
        scorer = RankingScorer(self._make_config())
        candidates = [
            _make_scored_candidate(
                peptide="MYPEPTIDE",
                wildtype_peptide="MYPEPTIDX",
                gene="BRCA1",
                mutation_str="BRCA1_p.Ala1Gly",
                best_allele="HLA-B*57:01",
                binding_affinity_nm=123.0,
                wildtype_affinity_nm=456.0,
                processing_score=0.77,
                agretopicity=3.7,
                expression=88.0,
                vaf=0.42,
            )
        ]
        result = scorer.rank(candidates)
        c = result[0]
        assert c.peptide == "MYPEPTIDE"
        assert c.wildtype_peptide == "MYPEPTIDX"
        assert c.gene == "BRCA1"
        assert c.mutation == "BRCA1_p.Ala1Gly"
        assert c.best_allele == "HLA-B*57:01"
        assert c.binding_affinity_nm == pytest.approx(123.0)
        assert c.wildtype_affinity_nm == pytest.approx(456.0)
        assert c.processing_score == pytest.approx(0.77)
        assert c.expression == pytest.approx(88.0)
        assert c.vaf == pytest.approx(0.42)
