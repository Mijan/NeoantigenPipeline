"""Tests for MHC-I binding prediction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from neoantigen_pipeline.candidates.peptide_generator import PeptideCandidate
from neoantigen_pipeline.config import MHCIPredictionConfig
from neoantigen_pipeline.exceptions import PredictionError
from neoantigen_pipeline.prediction.mhcflurry import MHCflurryPredictor
from neoantigen_pipeline.prediction.results import (
    BindingPrediction,
    MHCIPredictionResult,
    WildtypePredictionResult,
)


def _make_config(
    alleles: tuple[str, ...] = ("HLA-A*02:01",),
    lengths: tuple[int, ...] = (9,),
) -> MHCIPredictionConfig:
    return MHCIPredictionConfig(alleles=alleles, peptide_lengths=lengths)


def _make_candidate(
    peptide: str = "SIINFEKLV",
    wildtype: str = "SIINFEKLA",
    gene: str = "OVA",
    n_flank: str = "AAAA",
    c_flank: str = "BBBB",
) -> PeptideCandidate:
    return PeptideCandidate(
        peptide_sequence=peptide,
        wildtype_sequence=wildtype,
        gene=gene,
        mutation_str=f"{gene}_p.Val9Ala",
        transcript_id="ENST999",
        aa_pos=9,
        position_in_peptide=8,
        n_flank=n_flank,
        c_flank=c_flank,
        peptide_length=len(peptide),
    )


def _make_mhcflurry_result(peptides: list[str], allele: str = "HLA-A*02:01") -> pd.DataFrame:
    """Build a DataFrame matching MHCflurry's standardised output (one row per peptide)."""
    return pd.DataFrame(
        [
            {
                "peptide": p,
                "best_allele": allele,
                "mhcflurry_affinity": 150.0,
                "mhcflurry_affinity_percentile": 0.5,
                "mhcflurry_processing_score": 0.8,
                "mhcflurry_presentation_score": 0.7,
                "presentation_percentile": 1.5,
                "peptide_num": i,
                "sample_name": "sample1",
            }
            for i, p in enumerate(peptides)
        ]
    )


class TestMHCflurryPredictor:
    """Tests for the MHCflurryPredictor wrapper."""

    def test_name_property(self):
        predictor = MHCflurryPredictor(_make_config())
        assert predictor.name == "MHCflurry2-Class1Presentation"

    def test_mhc_class_property(self):
        predictor = MHCflurryPredictor(_make_config())
        assert predictor.mhc_class == 1

    def test_predict_returns_binding_predictions(self):
        predictor = MHCflurryPredictor(_make_config())
        predictor._predictor = MagicMock(
            predict=MagicMock(return_value=_make_mhcflurry_result(["SIINFEKLV"]))
        )

        results = predictor.predict(["SIINFEKLV"], ["HLA-A*02:01"])

        assert len(results) == 1
        assert isinstance(results[0], BindingPrediction)
        assert results[0].peptide == "SIINFEKLV"
        assert results[0].affinity_nm == pytest.approx(150.0)
        assert results[0].presentation_score == pytest.approx(0.7)

    def test_predict_calls_mhcflurry_with_correct_args(self):
        predictor = MHCflurryPredictor(_make_config())
        mock_backend = MagicMock(
            predict=MagicMock(return_value=_make_mhcflurry_result(["SIINFEKLV"]))
        )
        predictor._predictor = mock_backend

        predictor.predict(["SIINFEKLV"], ["HLA-A*02:01"])

        mock_backend.predict.assert_called_once_with(
            peptides=["SIINFEKLV"],
            alleles=["HLA-A*02:01"],
        )

    def test_predict_with_processing_returns_full_results(self):
        predictor = MHCflurryPredictor(_make_config())
        predictor._predictor = MagicMock(
            predict=MagicMock(return_value=_make_mhcflurry_result(["SIINFEKLV"]))
        )

        candidate = _make_candidate(gene="TESTGENE")
        results = predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, MHCIPredictionResult)
        assert result.peptide == "SIINFEKLV"
        assert result.gene == "TESTGENE"
        assert result.mutation_str == "TESTGENE_p.Val9Ala"
        assert result.affinity_nm == pytest.approx(150.0)
        assert result.presentation_score == pytest.approx(0.7)
        assert result.n_flank == "AAAA"
        assert result.c_flank == "BBBB"

    def test_predict_with_processing_passes_flanks_to_backend(self):
        predictor = MHCflurryPredictor(_make_config())
        mock_backend = MagicMock(
            predict=MagicMock(return_value=_make_mhcflurry_result(["SIINFEKLV"]))
        )
        predictor._predictor = mock_backend

        candidate = _make_candidate(n_flank="AAAA", c_flank="BBBB")
        predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        call_kwargs = mock_backend.predict.call_args.kwargs
        assert call_kwargs["n_flanks"] == ["AAAA"]
        assert call_kwargs["c_flanks"] == ["BBBB"]

    def test_predict_wildtype_returns_wildtype_results(self):
        predictor = MHCflurryPredictor(_make_config())
        predictor._predictor = MagicMock(
            predict=MagicMock(return_value=_make_mhcflurry_result(["SIINFEKLA"]))
        )

        candidate = _make_candidate(peptide="SIINFEKLV", wildtype="SIINFEKLA")
        results = predictor.predict_wildtype([candidate], ["HLA-A*02:01"])

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, WildtypePredictionResult)
        assert result.wt_peptide == "SIINFEKLA"
        assert result.mut_peptide == "SIINFEKLV"
        assert result.wildtype_affinity_nm == pytest.approx(150.0)

    def test_predict_with_processing_empty_returns_empty(self):
        predictor = MHCflurryPredictor(_make_config())
        assert predictor.predict_with_processing([], ["HLA-A*02:01"]) == []

    def test_predict_wildtype_empty_returns_empty(self):
        predictor = MHCflurryPredictor(_make_config())
        assert predictor.predict_wildtype([], ["HLA-A*02:01"]) == []

    def test_load_predictor_raises_on_missing_mhcflurry(self):
        predictor = MHCflurryPredictor(_make_config())
        with patch.dict("sys.modules", {"mhcflurry": None}):
            with pytest.raises(PredictionError, match="mhcflurry is not installed"):
                predictor._load_predictor()

    def test_standardise_columns_renames_correctly(self):
        predictor = MHCflurryPredictor(_make_config())
        raw_df = pd.DataFrame(
            {
                "peptide": ["SIINFEKLV"],
                "best_allele": ["HLA-A*02:01"],
                "affinity": [100.0],
                "affinity_percentile": [0.3],
                "processing_score": [0.9],
                "presentation_score": [0.85],
            }
        )
        result = predictor._standardise_columns(raw_df)
        assert "mhcflurry_affinity" in result.columns
        assert "mhcflurry_processing_score" in result.columns
        assert "mhcflurry_presentation_score" in result.columns
        assert "affinity" not in result.columns