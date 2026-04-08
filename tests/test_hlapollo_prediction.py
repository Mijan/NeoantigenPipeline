"""Tests for the HLApollo MHC-I binding predictor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from neoantigen_pipeline.candidates.peptide_generator import PeptideCandidate
from neoantigen_pipeline.exceptions import PredictorNotInstalledError
from neoantigen_pipeline.prediction.hlapollo import (
    HLApolloPredictor,
    _add_hla_prefix,
    _pad_c_flank,
    _pad_n_flank,
    _strip_hla_prefix,
)
from neoantigen_pipeline.prediction.results import (
    BindingPrediction,
    MHCIPredictionResult,
    WildtypePredictionResult,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_predictor(**kwargs) -> HLApolloPredictor:
    return HLApolloPredictor(
        binary_path="/fake/HLA-Apollo",
        **kwargs,
    )


def _make_candidate(
    peptide: str = "SIINFEKLV",
    wildtype: str = "SIINFEKLA",
    gene: str = "OVA",
    n_flank: str = "AAAAAA",
    c_flank: str = "BBBBBB",
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


def _make_hlapollo_output(
    row_idx: int = 0,
    allele: str = "A*02:01",
    peptide: str = "SIINFEKLV",
    score: float = 2.5,
    rank: float = 0.8,
) -> pd.DataFrame:
    """Build a minimal HLApollo output DataFrame."""
    return pd.DataFrame(
        [
            {
                "row_idx": row_idx,
                "allele": allele,
                "peptide": peptide,
                "n_flank": "****AAAAAA",
                "c_flank": "BBBBBB****",
                "invalid_allele": False,
                "invalid_aa": False,
                "invalid_length": False,
                "train_allele": True,
                "peptide_length": len(peptide),
                "mhc_pred_0": score,
                "mhc_pred_0_rank": rank,
            }
        ]
    )


# ── Allele normalization ──────────────────────────────────────────────────────


class TestAlleleNormalization:
    def test_strip_hla_prefix(self):
        assert _strip_hla_prefix("HLA-A*02:01") == "A*02:01"

    def test_strip_hla_prefix_already_stripped(self):
        assert _strip_hla_prefix("A*02:01") == "A*02:01"

    def test_add_hla_prefix(self):
        assert _add_hla_prefix("A*02:01") == "HLA-A*02:01"

    def test_add_hla_prefix_already_present(self):
        assert _add_hla_prefix("HLA-A*02:01") == "HLA-A*02:01"


# ── Flank padding ────────────────────────────────────────────────────────────


class TestFlankPadding:
    def test_n_flank_padded_left(self):
        assert _pad_n_flank("AAAAAA") == "****AAAAAA"

    def test_n_flank_exact_length(self):
        assert _pad_n_flank("AAAAAAAAAA") == "AAAAAAAAAA"

    def test_n_flank_empty(self):
        assert _pad_n_flank("") == "**********"

    def test_c_flank_padded_right(self):
        assert _pad_c_flank("BBBBBB") == "BBBBBB****"

    def test_c_flank_exact_length(self):
        assert _pad_c_flank("BBBBBBBBBB") == "BBBBBBBBBB"

    def test_c_flank_empty(self):
        assert _pad_c_flank("") == "**********"


# ── Properties ───────────────────────────────────────────────────────────────


class TestHLApolloProperties:
    def test_name(self):
        assert _make_predictor().name == "HLApollo"

    def test_mhc_class(self):
        assert _make_predictor().mhc_class == 1


# ── Binary check ─────────────────────────────────────────────────────────────


class TestBinaryCheck:
    def test_raises_when_binary_missing(self, tmp_path):
        predictor = HLApolloPredictor(
            binary_path=str(tmp_path / "nonexistent"),
            docker_image=None,
        )
        with pytest.raises(
            PredictorNotInstalledError, match="HLApollo binary not found"
        ):
            predictor._check_binary()

    def test_raises_when_docker_image_missing(self, tmp_path):
        predictor = HLApolloPredictor(
            binary_path=str(tmp_path / "nonexistent"),
            docker_image="hla-apollo",
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            with pytest.raises(
                PredictorNotInstalledError, match="Docker image 'hla-apollo' not found"
            ):
                predictor._check_binary()

    def test_no_raise_when_docker_image_found(self, tmp_path):
        predictor = HLApolloPredictor(
            binary_path=str(tmp_path / "nonexistent"),
            docker_image="hla-apollo",
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            predictor._check_binary()  # should not raise


# ── Input DataFrame construction ─────────────────────────────────────────────


class TestBuildInputDf:
    def test_combinations_generated(self):
        predictor = _make_predictor()
        df = predictor._build_input_df(
            peptides=["SIINFEKLV", "GILGFVFTL"],
            alleles=["HLA-A*02:01", "HLA-B*07:02"],
            n_flanks=["AAAA", "CCCC"],
            c_flanks=["BBBB", "DDDD"],
        )
        assert len(df) == 4  # 2 peptides × 2 alleles

    def test_allele_prefix_stripped(self):
        predictor = _make_predictor()
        df = predictor._build_input_df(
            peptides=["SIINFEKLV"],
            alleles=["HLA-A*02:01"],
            n_flanks=["AAAA"],
            c_flanks=["BBBB"],
        )
        assert df["allele"].iloc[0] == "A*02:01"

    def test_flanks_padded(self):
        predictor = _make_predictor()
        df = predictor._build_input_df(
            peptides=["SIINFEKLV"],
            alleles=["HLA-A*02:01"],
            n_flanks=["AAAA"],
            c_flanks=["BBBB"],
        )
        assert df["n_flank"].iloc[0] == "******AAAA"
        assert df["c_flank"].iloc[0] == "BBBB******"

    def test_row_idx_assigned_correctly(self):
        predictor = _make_predictor()
        df = predictor._build_input_df(
            peptides=["SIINFEKLV", "GILGFVFTL"],
            alleles=["HLA-A*02:01"],
            n_flanks=["AAAA", "CCCC"],
            c_flanks=["BBBB", "DDDD"],
        )
        assert set(df["row_idx"]) == {0, 1}


# ── Filter invalid ────────────────────────────────────────────────────────────


class TestFilterInvalid:
    def test_invalid_allele_filtered(self):
        predictor = _make_predictor()
        df = _make_hlapollo_output()
        df["invalid_allele"] = True
        result = predictor._filter_invalid(df)
        assert result.empty

    def test_invalid_aa_filtered(self):
        predictor = _make_predictor()
        df = _make_hlapollo_output()
        df["invalid_aa"] = True
        result = predictor._filter_invalid(df)
        assert result.empty

    def test_valid_rows_retained(self):
        predictor = _make_predictor()
        df = _make_hlapollo_output()
        result = predictor._filter_invalid(df)
        assert len(result) == 1


# ── Best allele selection ─────────────────────────────────────────────────────


class TestSelectBestAllele:
    def test_lowest_rank_selected(self):
        predictor = _make_predictor()
        df = pd.DataFrame(
            [
                {
                    "row_idx": 0,
                    "allele": "A*02:01",
                    "peptide": "SIINFEKLV",
                    "mhc_pred_0": 1.0,
                    "mhc_pred_0_rank": 2.0,
                },
                {
                    "row_idx": 0,
                    "allele": "B*07:02",
                    "peptide": "SIINFEKLV",
                    "mhc_pred_0": 3.0,
                    "mhc_pred_0_rank": 0.5,
                },
            ]
        )
        best = predictor._select_best_allele(df)
        assert len(best) == 1
        assert best.iloc[0]["allele"] == "B*07:02"

    def test_one_row_per_peptide_index(self):
        predictor = _make_predictor()
        df = pd.DataFrame(
            [
                {
                    "row_idx": 0,
                    "allele": "A*02:01",
                    "peptide": "P1",
                    "mhc_pred_0": 1.0,
                    "mhc_pred_0_rank": 1.0,
                },
                {
                    "row_idx": 0,
                    "allele": "B*07:02",
                    "peptide": "P1",
                    "mhc_pred_0": 2.0,
                    "mhc_pred_0_rank": 0.3,
                },
                {
                    "row_idx": 1,
                    "allele": "A*02:01",
                    "peptide": "P2",
                    "mhc_pred_0": 0.5,
                    "mhc_pred_0_rank": 5.0,
                },
            ]
        )
        best = predictor._select_best_allele(df)
        assert len(best) == 2
        assert set(best["row_idx"]) == {0, 1}


# ── predict() ────────────────────────────────────────────────────────────────


class TestPredict:
    def _mock_run(self, predictor, output_df):
        predictor._run_hlapollo = MagicMock(return_value=output_df)

    def test_returns_binding_predictions(self):
        predictor = _make_predictor()
        out = _make_hlapollo_output(row_idx=0, score=2.5, rank=0.8)
        self._mock_run(predictor, out)

        results = predictor.predict(["SIINFEKLV"], ["HLA-A*02:01"])
        assert len(results) == 1
        assert isinstance(results[0], BindingPrediction)
        assert results[0].peptide == "SIINFEKLV"
        assert results[0].presentation_score == pytest.approx(2.5)
        assert results[0].presentation_percentile == pytest.approx(0.8)

    def test_allele_prefix_added_to_result(self):
        predictor = _make_predictor()
        out = _make_hlapollo_output(allele="A*02:01")
        self._mock_run(predictor, out)

        results = predictor.predict(["SIINFEKLV"], ["HLA-A*02:01"])
        assert results[0].best_allele == "HLA-A*02:01"

    def test_empty_peptides_returns_empty(self):
        predictor = _make_predictor()
        assert predictor.predict([], ["HLA-A*02:01"]) == []


# ── predict_with_processing() ────────────────────────────────────────────────


class TestPredictWithProcessing:
    def _mock_run(self, predictor, output_df):
        predictor._run_hlapollo = MagicMock(return_value=output_df)

    def test_returns_mhci_prediction_results(self):
        predictor = _make_predictor()
        out = _make_hlapollo_output(score=3.1, rank=0.4)
        self._mock_run(predictor, out)

        candidate = _make_candidate(gene="BRCA1")
        results = predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        assert len(results) == 1
        assert isinstance(results[0], MHCIPredictionResult)
        assert results[0].gene == "BRCA1"
        assert results[0].presentation_score == pytest.approx(3.1)
        assert results[0].presentation_percentile == pytest.approx(0.4)

    def test_candidate_metadata_preserved(self):
        predictor = _make_predictor()
        out = _make_hlapollo_output()
        self._mock_run(predictor, out)

        candidate = _make_candidate(gene="TP53", n_flank="AAAA", c_flank="BBBB")
        results = predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        assert results[0].mutation_str == "TP53_p.Val9Ala"
        assert results[0].n_flank == "AAAA"
        assert results[0].c_flank == "BBBB"

    def test_flanks_passed_to_run(self):
        predictor = _make_predictor()
        predictor._check_binary = MagicMock()

        # Capture what _build_input_df receives
        original_build = predictor._build_input_df
        captured = {}

        def capturing_build(peptides, alleles, n_flanks, c_flanks):
            captured["n_flanks"] = n_flanks
            captured["c_flanks"] = c_flanks
            return original_build(peptides, alleles, n_flanks, c_flanks)

        predictor._build_input_df = capturing_build

        # Patch _run_chunk to return a fake output without running the binary
        fake_output = _make_hlapollo_output()
        predictor._run_chunk = MagicMock(return_value=fake_output)

        candidate = _make_candidate(n_flank="AAAA", c_flank="BBBB")
        predictor.predict_with_processing([candidate], ["HLA-A*02:01"])

        assert captured["n_flanks"] == ["AAAA"]
        assert captured["c_flanks"] == ["BBBB"]

    def test_empty_candidates_returns_empty(self):
        predictor = _make_predictor()
        assert predictor.predict_with_processing([], ["HLA-A*02:01"]) == []


# ── predict_wildtype() ────────────────────────────────────────────────────────


class TestPredictWildtype:
    def _mock_run(self, predictor, output_df):
        predictor._run_hlapollo = MagicMock(return_value=output_df)

    def test_returns_wildtype_results(self):
        predictor = _make_predictor()
        out = _make_hlapollo_output(peptide="SIINFEKLA", score=1.2, rank=3.5)
        self._mock_run(predictor, out)

        candidate = _make_candidate(peptide="SIINFEKLV", wildtype="SIINFEKLA")
        results = predictor.predict_wildtype([candidate], ["HLA-A*02:01"])

        assert len(results) == 1
        assert isinstance(results[0], WildtypePredictionResult)
        assert results[0].wt_peptide == "SIINFEKLA"
        assert results[0].mut_peptide == "SIINFEKLV"

    def test_empty_candidates_returns_empty(self):
        predictor = _make_predictor()
        assert predictor.predict_wildtype([], ["HLA-A*02:01"]) == []


# ── Integration test (skipped if binary absent) ───────────────────────────────


@pytest.mark.integration
def test_hlapollo_integration_real_binary():
    binary = Path("tools/HLApollo/HLA-Apollo")
    if not binary.is_file():
        pytest.skip("HLApollo binary not installed at tools/HLApollo/HLA-Apollo")

    predictor = HLApolloPredictor(binary_path=str(binary), docker_image=None)
    results = predictor.predict(
        peptides=["SIINFEKLV", "GILGFVFTL"],
        alleles=["HLA-A*02:01"],
    )
    assert len(results) == 2
    for r in results:
        assert isinstance(r, BindingPrediction)
        assert r.presentation_percentile >= 0.0


@pytest.mark.integration
def test_hlapollo_docker_prediction():
    """Run HLApollo via Docker on a small set of peptides."""
    predictor = HLApolloPredictor(docker_image="hla-apollo")
    results = predictor.predict(
        peptides=["SIINFEKL", "GILGFVFTL"],
        alleles=["HLA-A*02:01"],
    )
    assert len(results) == 2
    assert all(r.presentation_percentile < 100.0 for r in results)
