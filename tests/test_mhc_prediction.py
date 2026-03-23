"""Tests for MHC-I binding prediction wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from neoantigen_pipeline.config import MHCIPredictionConfig
from neoantigen_pipeline.exceptions import PredictionError
from neoantigen_pipeline.prediction.mhc_i import MHCIPredictor
from neoantigen_pipeline.processing.peptide_generator import PeptideCandidate


def _make_config(
    alleles=("HLA-A*02:01",),
    lengths=(9,),
) -> MHCIPredictionConfig:
    return MHCIPredictionConfig(alleles=alleles, peptide_lengths=lengths)


def _make_candidate(
    peptide: str = "SIINFEKLV",
    wildtype: str = "SIINFEKLV",
    gene: str = "OVA",
    n_flank: str = "AAAA",
    c_flank: str = "BBBB",
) -> PeptideCandidate:
    return PeptideCandidate(
        peptide_sequence=peptide,
        wildtype_sequence=wildtype,
        gene=gene,
        mutation_str=f"{gene}_p.X9Y",
        transcript_id="ENST999",
        aa_pos=9,
        position_in_peptide=4,
        n_flank=n_flank,
        c_flank=c_flank,
        peptide_length=len(peptide),
    )


def _make_mhcflurry_result(peptides, alleles) -> pd.DataFrame:
    """Build a DataFrame matching MHCflurry's raw output format."""
    rows = []
    for peptide in peptides:
        for allele in alleles:
            rows.append(
                {
                    "peptide": peptide,
                    "allele": allele,
                    "affinity": 150.0,
                    "affinity_percentile": 0.5,
                    "processing_score": 0.8,
                    "presentation_score": 0.7,
                }
            )
    return pd.DataFrame(rows)


class TestMHCIPredictor:
    """Tests for the MHCIPredictor wrapper."""

    def test_name_property(self):
        config = _make_config()
        predictor = MHCIPredictor(config)
        assert predictor.name == "MHCflurry2-Class1Presentation"

    def test_mhc_class_property(self):
        config = _make_config()
        predictor = MHCIPredictor(config)
        assert predictor.mhc_class == 1

    def test_predict_calls_mhcflurry(self):
        config = _make_config()
        predictor = MHCIPredictor(config)

        mock_flurry = MagicMock()
        mock_flurry.predict.return_value = _make_mhcflurry_result(
            ["SIINFEKLV"], ["HLA-A*02:01"]
        )
        predictor._predictor = mock_flurry

        result = predictor.predict(["SIINFEKLV"], ["HLA-A*02:01"])

        mock_flurry.predict.assert_called_once_with(
            peptides=["SIINFEKLV"],
            alleles=["HLA-A*02:01"],
        )
        assert "mhcflurry_affinity" in result.columns
        assert "mhcflurry_presentation_score" in result.columns

    def test_predict_with_processing_includes_flanks(self):
        config = _make_config()
        predictor = MHCIPredictor(config)

        mock_flurry = MagicMock()
        mock_flurry.predict.return_value = _make_mhcflurry_result(
            ["SIINFEKLV"], ["HLA-A*02:01"]
        )
        predictor._predictor = mock_flurry

        candidate = _make_candidate()
        result = predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        call_kwargs = mock_flurry.predict.call_args
        assert call_kwargs.kwargs.get("n_flanks") is not None or (
            call_kwargs.args and len(call_kwargs.args) >= 3
        )

        assert not result.empty

    def test_predict_with_processing_attaches_metadata(self):
        config = _make_config()
        predictor = MHCIPredictor(config)

        mock_flurry = MagicMock()
        mock_flurry.predict.return_value = _make_mhcflurry_result(
            ["SIINFEKLV"], ["HLA-A*02:01"]
        )
        predictor._predictor = mock_flurry

        candidate = _make_candidate(gene="TESTGENE")
        result = predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        assert "gene" in result.columns
        assert result["gene"].iloc[0] == "TESTGENE"

    def test_predict_wildtype_renames_affinity(self):
        config = _make_config()
        predictor = MHCIPredictor(config)

        mock_flurry = MagicMock()
        mock_flurry.predict.return_value = _make_mhcflurry_result(
            ["SIINFEKLV"], ["HLA-A*02:01"]
        )
        predictor._predictor = mock_flurry

        candidate = _make_candidate()
        result = predictor.predict_wildtype([candidate], ["HLA-A*02:01"])

        assert "wildtype_affinity" in result.columns
        assert "mhcflurry_affinity" not in result.columns

    def test_predict_empty_candidates_returns_empty_df(self):
        config = _make_config()
        predictor = MHCIPredictor(config)
        result = predictor.predict_with_processing([], ["HLA-A*02:01"])
        assert result.empty

    def test_load_predictor_raises_prediction_error_on_import_fail(self):
        config = _make_config()
        predictor = MHCIPredictor(config)

        with patch.dict("sys.modules", {"mhcflurry": None}):
            with pytest.raises(PredictionError, match="mhcflurry is not installed"):
                predictor._load_predictor()

    def test_standardise_columns_renames_correctly(self):
        config = _make_config()
        predictor = MHCIPredictor(config)

        raw_df = pd.DataFrame(
            {
                "peptide": ["SIINFEKLV"],
                "allele": ["HLA-A*02:01"],
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
