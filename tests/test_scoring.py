"""Tests for agretopicity, foreignness, and ranking scorers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neoantigen_pipeline.config import ScoringConfig
from neoantigen_pipeline.scoring.agretopicity import AgretopicityScorer
from neoantigen_pipeline.scoring.foreignness import ForeignnessScorer
from neoantigen_pipeline.scoring.ranking import RankingScorer


class TestAgretopicityScorer:
    """Tests for AgretopicityScorer."""

    def setup_method(self):
        self.scorer = AgretopicityScorer()

    def test_basic_computation(self):
        # wildtype binds at 1000 nM, mutant at 100 nM => agretopicity = 10
        result = self.scorer.compute(100.0, 1000.0)
        assert result == pytest.approx(10.0)

    def test_equal_affinity_returns_one(self):
        result = self.scorer.compute(500.0, 500.0)
        assert result == pytest.approx(1.0)

    def test_mutant_affinity_zero_returns_zero(self):
        result = self.scorer.compute(0.0, 1000.0)
        assert result == 0.0

    def test_wildtype_stronger_returns_less_than_one(self):
        # Wildtype binds better (lower nM), so ratio < 1
        result = self.scorer.compute(500.0, 100.0)
        assert result == pytest.approx(0.2)

    def test_annotate_dataframe_adds_column(self):
        df = pd.DataFrame(
            {
                "peptide": ["SIINFEKLV", "GILGFVFTL"],
                "allele": ["HLA-A*02:01", "HLA-A*02:01"],
                "mhcflurry_affinity": [100.0, 500.0],
                "wildtype_affinity": [1000.0, 1000.0],
            }
        )
        result = self.scorer.annotate_dataframe(df)
        assert "agretopicity" in result.columns
        assert result["agretopicity"].iloc[0] == pytest.approx(10.0)
        assert result["agretopicity"].iloc[1] == pytest.approx(2.0)

    def test_annotate_dataframe_raises_on_missing_columns(self):
        df = pd.DataFrame({"peptide": ["SIINFEKLV"]})
        with pytest.raises(KeyError):
            self.scorer.annotate_dataframe(df)

    def test_annotate_dataframe_does_not_mutate_input(self):
        df = pd.DataFrame(
            {
                "mhcflurry_affinity": [100.0],
                "wildtype_affinity": [200.0],
            }
        )
        original_cols = set(df.columns)
        self.scorer.annotate_dataframe(df)
        assert set(df.columns) == original_cols


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
    """Tests for RankingScorer normalisation and weighting."""

    def _make_config(self, **kwargs) -> ScoringConfig:
        defaults = dict(
            presentation_score_weight=0.4,
            agretopicity_weight=0.2,
            expression_weight=0.2,
            vaf_weight=0.2,
        )
        defaults.update(kwargs)
        return ScoringConfig(**defaults)

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
                "allele": ["HLA-A*02:01"] * 3,
                "mhcflurry_presentation_score": [0.9, 0.5, 0.1],
                "agretopicity": [5.0, 2.0, 1.0],
                "expression": [100.0, 50.0, 10.0],
                "vaf": [0.5, 0.3, 0.1],
            }
        )

    def test_rank_adds_composite_score(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        df = self._make_df()
        result = scorer.rank(df)
        assert "composite_score" in result.columns

    def test_rank_adds_rank_column(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        df = self._make_df()
        result = scorer.rank(df)
        assert "composite_rank" in result.columns
        assert list(result["composite_rank"]) == [1, 2, 3]

    def test_rank_sorted_descending(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        df = self._make_df()
        result = scorer.rank(df)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_top_candidate_has_highest_scores(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        df = self._make_df()
        result = scorer.rank(df)
        # PEPTIDE1 should rank first as it has highest values in all columns
        assert result["peptide"].iloc[0] == "PEPTIDE1"

    def test_normalise_constant_column_returns_zeros(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        df = pd.DataFrame(
            {
                "mhcflurry_presentation_score": [0.5, 0.5, 0.5],
                "agretopicity": [1.0, 2.0, 3.0],
                "expression": [1.0, 2.0, 3.0],
                "vaf": [0.1, 0.2, 0.3],
            }
        )
        norm = scorer._normalise(df, "mhcflurry_presentation_score")
        assert np.all(norm == 0.0)

    def test_normalise_missing_column_returns_zeros_with_warning(self, caplog):
        import logging

        config = self._make_config()
        scorer = RankingScorer(config)
        df = pd.DataFrame({"other_col": [1.0, 2.0]})
        with caplog.at_level(logging.WARNING):
            norm = scorer._normalise(df, "missing_col")
        assert np.all(norm == 0.0)
        assert "missing_col" in caplog.text

    def test_rank_empty_dataframe_raises(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        with pytest.raises(ValueError):
            scorer.rank(pd.DataFrame())

    def test_composite_score_in_zero_one_range(self):
        config = self._make_config()
        scorer = RankingScorer(config)
        df = self._make_df()
        result = scorer.rank(df)
        assert result["composite_score"].between(0.0, 1.0).all()
