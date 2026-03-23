"""HLApollo-backed MHC class I binding predictor.

Wraps the HLApollo binary (Genentech, Thrift et al. Nature Communications 2024)
as a subprocess. Input peptides are written to a temporary CSV, the binary is
called, and output is parsed back into typed result objects.

HLApollo scores one peptide-allele pair at a time. The wrapper generates all
combinations, runs them (optionally in batches), and selects the best allele
per peptide by lowest percentile rank.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from neoantigen_pipeline.exceptions import PredictionError, PredictorNotInstalledError
from neoantigen_pipeline.prediction.mhc_i import MHCIPredictor
from neoantigen_pipeline.prediction.results import (
    BindingPrediction,
    MHCIPredictionResult,
    WildtypePredictionResult,
)

if TYPE_CHECKING:
    from neoantigen_pipeline.candidates.peptide_generator import PeptideCandidate

# Column names in HLApollo's CSV output
_COL_ALLELE = "allele"
_COL_PEPTIDE = "peptide"
_COL_N_FLANK = "n_flank"
_COL_C_FLANK = "c_flank"
_COL_INVALID_ALLELE = "invalid_allele"
_COL_INVALID_AA = "invalid_aa"
_COL_INVALID_LENGTH = "invalid_length"
_COL_SCORE = "mhc_pred_0"
_COL_RANK = "mhc_pred_0_rank"

# Internal tracking column (passed through HLApollo unchanged)
_COL_ROW_IDX = "row_idx"

# HLApollo required flank length
_FLANK_LENGTH = 10
_FLANK_PAD = "*"


def _pad_n_flank(flank: str) -> str:
    """Right-justify N-flank to exactly 10 chars, padding with '*' on the left."""
    return flank.rjust(_FLANK_LENGTH, _FLANK_PAD)


def _pad_c_flank(flank: str) -> str:
    """Left-justify C-flank to exactly 10 chars, padding with '*' on the right."""
    return flank.ljust(_FLANK_LENGTH, _FLANK_PAD)


def _strip_hla_prefix(allele: str) -> str:
    """Convert 'HLA-A*02:01' → 'A*02:01' as required by HLApollo."""
    return allele.removeprefix("HLA-")


def _add_hla_prefix(allele: str) -> str:
    """Convert 'A*02:01' → 'HLA-A*02:01' for pipeline consistency."""
    if allele.startswith("HLA-"):
        return allele
    return f"HLA-{allele}"


class HLApolloPredictor(MHCIPredictor):
    """MHC-I presentation predictor backed by the HLApollo binary.

    HLApollo is invoked as a subprocess. Input peptides are written to a
    temporary CSV, the binary is called, and output is parsed back into typed
    result objects.

    Multi-allele handling: HLApollo scores one peptide-allele pair at a time.
    This wrapper generates all combinations for a given genotype, runs them
    (optionally in batches), and selects the best allele per peptide by lowest
    percentile rank (``mhc_pred_0_rank``).

    Args:
        binary_path: Path to the HLA-Apollo executable.
        docker_image: If set, run via Docker instead of native binary. Pass
            the image name, e.g. ``"hla-apollo"``.
        timeout_seconds: Max seconds to wait for the binary to finish.
        batch_size: Number of peptide-allele pairs per subprocess call.
            Reduce if you hit memory limits.
    """

    def __init__(
        self,
        binary_path: str = "tools/HLApollo/HLA-Apollo",
        docker_image: str | None = None,
        timeout_seconds: int = 600,
        batch_size: int = 5000,
    ) -> None:
        self._binary_path = str(binary_path)
        self._docker_image = docker_image
        self._timeout_seconds = timeout_seconds
        self._batch_size = batch_size
        self._logger = logging.getLogger(type(self).__qualname__)

    # ── BindingPredictor interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        """Predictor name.

        Returns:
            "HLApollo"
        """
        return "HLApollo"

    def predict(
        self, peptides: list[str], alleles: list[str]
    ) -> list[BindingPrediction]:
        """Predict MHC-I presentation for peptides against a genotype.

        Generates all peptide x allele combinations, runs HLApollo, and
        returns the best-allele prediction per peptide.

        Args:
            peptides: Amino acid sequences.
            alleles: HLA allele strings in "HLA-A*02:01" notation.

        Returns:
            List of ``BindingPrediction`` objects, one per input peptide, in
            the same order as ``peptides``.

        Raises:
            PredictorNotInstalledError: If the HLApollo binary is not found.
            PredictionError: If the HLApollo binary fails.
        """
        if not peptides:
            return []

        # Use empty flanks; will be padded to '*' * 10
        n_flanks = [""] * len(peptides)
        c_flanks = [""] * len(peptides)
        df = self._run_hlapollo(peptides, alleles, n_flanks, c_flanks)
        best = self._select_best_allele(df)

        results = []
        for idx in range(len(peptides)):
            row = best[best[_COL_ROW_IDX] == idx]
            if row.empty:
                self._logger.warning(
                    "No valid HLApollo result for peptide %d ('%s')", idx, peptides[idx]
                )
                results.append(
                    BindingPrediction(
                        peptide=peptides[idx],
                        best_allele="",
                        affinity_nm=0.0,
                        affinity_percentile=0.0,
                        processing_score=0.0,
                        presentation_score=0.0,
                        presentation_percentile=100.0,
                    )
                )
            else:
                r = row.iloc[0]
                results.append(
                    BindingPrediction(
                        peptide=str(r[_COL_PEPTIDE]),
                        best_allele=_add_hla_prefix(str(r[_COL_ALLELE])),
                        affinity_nm=0.0,
                        affinity_percentile=0.0,
                        processing_score=0.0,
                        presentation_score=float(r[_COL_SCORE]),
                        presentation_percentile=float(r[_COL_RANK]),
                    )
                )
        return results

    # ── MHCIPredictor interface ──────────────────────────────────────────────

    def predict_with_processing(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> list[MHCIPredictionResult]:
        """Predict MHC-I presentation for candidates with flank context.

        Args:
            candidates: ``PeptideCandidate`` objects with sequences and flanks.
            alleles: HLA-I allele strings.

        Returns:
            List of ``MHCIPredictionResult`` objects, one per candidate, in
            the same order as ``candidates``.

        Raises:
            PredictorNotInstalledError: If the HLApollo binary is not found.
            PredictionError: If the HLApollo binary fails.
        """
        if not candidates:
            return []

        peptides = [c.peptide_sequence for c in candidates]
        n_flanks = [c.n_flank for c in candidates]
        c_flanks = [c.c_flank for c in candidates]
        df = self._run_hlapollo(peptides, alleles, n_flanks, c_flanks)
        best = self._select_best_allele(df)

        results = []
        for i, candidate in enumerate(candidates):
            row = best[best[_COL_ROW_IDX] == i]
            if row.empty:
                self._logger.warning(
                    "No valid HLApollo result for candidate %d ('%s')",
                    i,
                    candidate.peptide_sequence,
                )
                score, rank, best_allele = 0.0, 100.0, ""
            else:
                r = row.iloc[0]
                score = float(r[_COL_SCORE])
                rank = float(r[_COL_RANK])
                best_allele = _add_hla_prefix(str(r[_COL_ALLELE]))

            results.append(
                MHCIPredictionResult(
                    peptide=candidate.peptide_sequence,
                    best_allele=best_allele,
                    affinity_nm=0.0,
                    affinity_percentile=0.0,
                    processing_score=0.0,
                    presentation_score=score,
                    presentation_percentile=rank,
                    mutation_str=candidate.mutation_str,
                    transcript_id=candidate.transcript_id,
                    gene=candidate.gene,
                    aa_pos=candidate.aa_pos,
                    n_flank=candidate.n_flank,
                    c_flank=candidate.c_flank,
                )
            )
        return results

    def predict_wildtype(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> list[WildtypePredictionResult]:
        """Predict MHC-I binding for the wildtype counterpart of each candidate.

        Args:
            candidates: ``PeptideCandidate`` objects whose ``wildtype_sequence``
                is used as the query peptide.
            alleles: HLA-I allele strings.

        Returns:
            List of ``WildtypePredictionResult`` objects, one per candidate, in
            the same order as ``candidates``.

        Raises:
            PredictorNotInstalledError: If the HLApollo binary is not found.
            PredictionError: If the HLApollo binary fails.
        """
        if not candidates:
            return []

        wt_peptides = [c.wildtype_sequence for c in candidates]
        mut_peptides = [c.peptide_sequence for c in candidates]
        n_flanks = [c.n_flank for c in candidates]
        c_flanks = [c.c_flank for c in candidates]
        df = self._run_hlapollo(wt_peptides, alleles, n_flanks, c_flanks)
        best = self._select_best_allele(df)

        results = []
        for i, (wt_pep, mut_pep) in enumerate(zip(wt_peptides, mut_peptides)):
            row = best[best[_COL_ROW_IDX] == i]
            wt_affinity = float(row.iloc[0][_COL_SCORE]) if not row.empty else 0.0
            results.append(
                WildtypePredictionResult(
                    wt_peptide=wt_pep,
                    mut_peptide=mut_pep,
                    wildtype_affinity_nm=wt_affinity,
                )
            )
        return results

    # ── Private helpers ──────────────────────────────────────────────────────

    def _check_binary(self) -> None:
        """Verify the HLApollo binary exists.

        Raises:
            PredictorNotInstalledError: If the binary is not found at
                ``binary_path``.
        """
        if self._docker_image:
            return  # Docker: binary check not applicable
        if not Path(self._binary_path).is_file():
            raise PredictorNotInstalledError(
                f"HLApollo binary not found at '{self._binary_path}'. "
                "Install with:\n"
                "  mkdir -p tools && cd tools\n"
                "  git clone https://github.com/Genentech/HLApollo.git\n"
                "  chmod +x HLApollo/HLA-Apollo"
            )

    def _build_input_df(
        self,
        peptides: list[str],
        alleles: list[str],
        n_flanks: list[str],
        c_flanks: list[str],
    ) -> pd.DataFrame:
        """Build the all-combinations input DataFrame for HLApollo.

        For each (peptide, allele) pair, emits one row. The ``row_idx`` column
        tracks which original peptide index the row belongs to so results can
        be reassembled after best-allele selection.
        """
        rows = []
        for idx, (peptide, n_flank, c_flank) in enumerate(
            zip(peptides, n_flanks, c_flanks)
        ):
            for allele in alleles:
                rows.append(
                    {
                        _COL_ROW_IDX: idx,
                        _COL_ALLELE: _strip_hla_prefix(allele),
                        _COL_PEPTIDE: peptide,
                        _COL_N_FLANK: _pad_n_flank(n_flank),
                        _COL_C_FLANK: _pad_c_flank(c_flank),
                    }
                )
        return pd.DataFrame(rows)

    def _run_hlapollo(
        self,
        peptides: list[str],
        alleles: list[str],
        n_flanks: list[str],
        c_flanks: list[str],
    ) -> pd.DataFrame:
        """Run the HLApollo binary and return the raw output DataFrame.

        Splits the input into batches of ``batch_size``, runs each batch, and
        concatenates results. Invalid rows (invalid allele, amino acid, or
        length) are logged and filtered out.

        Args:
            peptides: Peptide sequences.
            alleles: HLA allele strings (pipeline format, with "HLA-" prefix).
            n_flanks: N-terminal flanking sequences per peptide.
            c_flanks: C-terminal flanking sequences per peptide.

        Returns:
            Concatenated output DataFrame with rows for all valid predictions.

        Raises:
            PredictorNotInstalledError: If the binary is missing.
            PredictionError: If the subprocess call fails.
        """
        self._check_binary()
        input_df = self._build_input_df(peptides, alleles, n_flanks, c_flanks)

        chunks = [
            input_df.iloc[i : i + self._batch_size]
            for i in range(0, len(input_df), self._batch_size)
        ]

        output_parts: list[pd.DataFrame] = []
        for chunk in chunks:
            output_parts.append(self._run_chunk(chunk))

        result = pd.concat(output_parts, ignore_index=True)
        return self._filter_invalid(result)

    def _run_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Run a single batch through the HLApollo binary.

        Args:
            chunk: Subset of the input DataFrame.

        Returns:
            Output DataFrame for this chunk.

        Raises:
            PredictionError: If the subprocess call fails.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "input.csv")
            out_path = os.path.join(tmpdir, "output.csv")
            chunk.to_csv(in_path, index=False)

            if self._docker_image:
                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{tmpdir}:/data",
                    self._docker_image,
                    "/home/HLA-Apollo/HLA-Apollo",
                    "/data/input.csv",
                    "/data/output.csv",
                ]
            else:
                cmd = [self._binary_path, in_path, out_path]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    timeout=self._timeout_seconds,
                    capture_output=True,
                )
            except FileNotFoundError as exc:
                raise PredictorNotInstalledError(
                    f"HLApollo binary not found: {exc}"
                ) from exc
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                raise PredictionError(
                    f"HLApollo subprocess failed (exit {exc.returncode}): {stderr}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise PredictionError(
                    f"HLApollo timed out after {self._timeout_seconds}s"
                ) from exc

            try:
                return pd.read_csv(out_path)
            except Exception as exc:
                raise PredictionError(
                    f"Failed to parse HLApollo output: {exc}"
                ) from exc

    def _filter_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log and remove rows with invalid allele, amino acid, or length flags."""
        invalid_mask = (
            df.get(_COL_INVALID_ALLELE, False).astype(bool)
            | df.get(_COL_INVALID_AA, False).astype(bool)
            | df.get(_COL_INVALID_LENGTH, False).astype(bool)
        )
        n_invalid = int(invalid_mask.sum())
        if n_invalid:
            invalid_rows = df[invalid_mask]
            self._logger.warning(
                "HLApollo flagged %d row(s) as invalid; they will be excluded. "
                "Breakdown — invalid_allele: %d, invalid_aa: %d, invalid_length: %d",
                n_invalid,
                int(invalid_rows.get(_COL_INVALID_ALLELE, pd.Series(dtype=bool)).sum()),
                int(invalid_rows.get(_COL_INVALID_AA, pd.Series(dtype=bool)).sum()),
                int(invalid_rows.get(_COL_INVALID_LENGTH, pd.Series(dtype=bool)).sum()),
            )
        return df[~invalid_mask].reset_index(drop=True)

    def _select_best_allele(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each original peptide index, keep the row with the lowest rank.

        Groups by ``row_idx`` and retains the row with the minimum
        ``mhc_pred_0_rank`` (lowest percentile rank = best presenter).

        Args:
            df: Filtered HLApollo output (may contain multiple alleles per
                peptide index).

        Returns:
            DataFrame with at most one row per ``row_idx``.
        """
        if df.empty or _COL_ROW_IDX not in df.columns:
            return df
        idx = df.groupby(_COL_ROW_IDX)[_COL_RANK].idxmin()
        return df.loc[idx].reset_index(drop=True)
