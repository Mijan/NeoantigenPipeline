"""MHCflurry-backed MHC class I binding predictor.

Wraps the MHCflurry ``Class1PresentationPredictor`` and adapts its raw
DataFrame output into typed ``MHCIPredictionResult`` and
``WildtypePredictionResult`` objects. The predictor is lazy-loaded on
first use.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neoantigen_pipeline._constants import (
    COL_AFFINITY_PERCENTILE,
    COL_BEST_ALLELE,
    COL_MHCFLURRY_AFFINITY,
    COL_PEPTIDE,
    COL_PEPTIDE_NUM,
    COL_PRESENTATION_PERCENTILE,
    COL_PRESENTATION_SCORE,
    COL_PROCESSING_SCORE,
    MHCFLURRY_RENAME_MAP,
)
from neoantigen_pipeline.exceptions import PredictionError
from neoantigen_pipeline.prediction.mhc_i import MHCIPredictor
from neoantigen_pipeline.prediction.results import (
    BindingPrediction,
    MHCIPredictionResult,
    WildtypePredictionResult,
)

if TYPE_CHECKING:
    import pandas as pd

    from neoantigen_pipeline.candidates.peptide_generator import PeptideCandidate
    from neoantigen_pipeline.config import MHCIPredictionConfig

# Default fallback values for optional MHCflurry output columns
_DEFAULT_AFFINITY_PERCENTILE: float = 0.0
_DEFAULT_PROCESSING_SCORE: float = 0.0
_DEFAULT_PRESENTATION_PERCENTILE: float = 0.0


class MHCflurryPredictor(MHCIPredictor):
    """MHC class I binding predictor backed by MHCflurry 2.x.

    Uses the MHCflurry ``Class1PresentationPredictor`` which jointly predicts
    MHC-I binding affinity, antigen processing, and a combined presentation
    score for each peptide–allele pair.

    The underlying MHCflurry model is loaded lazily on first use to avoid
    import overhead when the class is instantiated.

    Args:
        config: MHC-I prediction configuration specifying alleles, peptide
            lengths, score thresholds, and presentation-score usage.

    Raises:
        PredictionError: If the MHCflurry models cannot be loaded.
    """

    def __init__(self, config: MHCIPredictionConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(type(self).__qualname__)
        self._predictor: Any = None  # Lazy-loaded

    # ── BindingPredictor interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        """Predictor name.

        Returns:
            "MHCflurry2-Class1Presentation"
        """
        return "MHCflurry2-Class1Presentation"

    def predict(
        self, peptides: list[str], alleles: list[str]
    ) -> list[BindingPrediction]:
        """Predict MHC-I binding without candidate-level metadata.

        Suitable for simple queries where only peptide sequences and alleles
        are available. For pipeline use with ``PeptideCandidate`` objects,
        prefer ``predict_with_processing`` to include processing context.

        Args:
            peptides: List of amino acid sequences.
            alleles: List of HLA allele strings.

        Returns:
            List of ``BindingPrediction`` objects, one per input peptide.

        Raises:
            PredictionError: If prediction fails.
        """
        raw_df = self._run_mhcflurry(peptides=peptides, alleles=alleles)
        return self._df_to_binding_predictions(raw_df, peptides)

    # ── MHCIPredictor interface ──────────────────────────────────────────────

    def predict_with_processing(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> list[MHCIPredictionResult]:
        """Predict MHC-I presentation for peptide candidates with flank context.

        Passes N- and C-terminal flanking sequences to MHCflurry's antigen
        processing model for improved prediction accuracy.

        Args:
            candidates: ``PeptideCandidate`` objects with sequences and flanks.
            alleles: HLA-I allele strings to predict against.

        Returns:
            List of ``MHCIPredictionResult`` objects, one per candidate,
            in the same order as ``candidates``.

        Raises:
            PredictionError: If prediction fails.
        """
        if not candidates:
            return []

        peptides = [c.peptide_sequence for c in candidates]
        raw_df = self._run_mhcflurry(
            peptides=peptides,
            alleles=alleles,
            n_flanks=[c.n_flank for c in candidates],
            c_flanks=[c.c_flank for c in candidates],
        )
        return self._df_to_prediction_results(raw_df, candidates)

    def predict_wildtype(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> list[WildtypePredictionResult]:
        """Predict MHC-I binding for the wildtype counterpart of each candidate.

        The result at index *i* is paired with the ``predict_with_processing``
        result at index *i* — both derive from ``candidates[i]``.

        Args:
            candidates: ``PeptideCandidate`` objects whose ``wildtype_sequence``
                is used as the query peptide.
            alleles: HLA-I allele strings.

        Returns:
            List of ``WildtypePredictionResult`` objects, one per candidate,
            in the same order as ``candidates``.

        Raises:
            PredictionError: If prediction fails.
        """
        if not candidates:
            return []

        wt_peptides = [c.wildtype_sequence for c in candidates]
        mut_peptides = [c.peptide_sequence for c in candidates]

        raw_df = self._run_mhcflurry(
            peptides=wt_peptides,
            alleles=alleles,
            n_flanks=[c.n_flank for c in candidates],
            c_flanks=[c.c_flank for c in candidates],
        )
        return self._df_to_wildtype_results(raw_df, wt_peptides, mut_peptides)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_predictor(self) -> Any:
        """Load and cache the MHCflurry ``Class1PresentationPredictor``.

        Returns:
            Loaded MHCflurry predictor instance.

        Raises:
            PredictionError: If MHCflurry or its models cannot be loaded.
        """
        if self._predictor is not None:
            return self._predictor

        try:
            from mhcflurry import Class1PresentationPredictor

            self._logger.info("Loading MHCflurry Class1PresentationPredictor...")
            self._predictor = Class1PresentationPredictor.load()
            self._logger.info("MHCflurry predictor loaded successfully")
        except ImportError as exc:
            if "pkg_resources" in str(exc):
                raise PredictionError(
                    "mhcflurry requires 'pkg_resources', which was removed in "
                    "setuptools>=71. Fix with: pip install 'setuptools<71'"
                ) from exc
            raise PredictionError(
                "mhcflurry is not installed. "
                "Install with:\n"
                "  pip install mhcflurry\n"
                "  mhcflurry-downloads fetch"
            ) from exc
        except Exception as exc:
            raise PredictionError(f"Failed to load MHCflurry models: {exc}") from exc

        return self._predictor

    def _run_mhcflurry(
        self,
        peptides: list[str],
        alleles: list[str],
        n_flanks: list[str] | None = None,
        c_flanks: list[str] | None = None,
    ) -> pd.DataFrame:
        """Call MHCflurry and return a standardised DataFrame.

        This is the only place in the class that touches a DataFrame directly;
        all callers immediately convert to typed result objects.

        Args:
            peptides: Peptide sequences (one per candidate).
            alleles: HLA alleles to predict against.
            n_flanks: Optional N-terminal flanking sequences.
            c_flanks: Optional C-terminal flanking sequences.

        Returns:
            Standardised MHCflurry output with renamed columns.

        Raises:
            PredictionError: If MHCflurry raises an error.
        """
        predictor = self._load_predictor()
        kwargs: dict[str, list[str]] = {}
        if n_flanks is not None:
            kwargs["n_flanks"] = n_flanks
        if c_flanks is not None:
            kwargs["c_flanks"] = c_flanks

        try:
            result: pd.DataFrame = predictor.predict(
                peptides=peptides,
                alleles=alleles,
                **kwargs,
            )
        except Exception as exc:
            raise PredictionError(f"MHCflurry prediction failed: {exc}") from exc

        return self._standardise_columns(result)

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename MHCflurry output columns to pipeline-standard names.

        Args:
            df: Raw MHCflurry prediction DataFrame.

        Returns:
            DataFrame with standardised column names per ``MHCFLURRY_RENAME_MAP``.
        """
        actual_renames = {
            k: v for k, v in MHCFLURRY_RENAME_MAP.items() if k in df.columns
        }
        return df.rename(columns=actual_renames)

    def _row_to_binding_prediction(self, row: Any, peptide: str) -> BindingPrediction:
        """Extract a ``BindingPrediction`` from a standardised DataFrame row."""
        return BindingPrediction(
            peptide=str(row.get(COL_PEPTIDE, peptide)),
            best_allele=str(row.get(COL_BEST_ALLELE, "")),
            affinity_nm=float(row.get(COL_MHCFLURRY_AFFINITY, 0.0)),
            affinity_percentile=float(
                row.get(COL_AFFINITY_PERCENTILE, _DEFAULT_AFFINITY_PERCENTILE)
            ),
            processing_score=float(
                row.get(COL_PROCESSING_SCORE, _DEFAULT_PROCESSING_SCORE)
            ),
            presentation_score=float(row.get(COL_PRESENTATION_SCORE, 0.0)),
            presentation_percentile=float(
                row.get(COL_PRESENTATION_PERCENTILE, _DEFAULT_PRESENTATION_PERCENTILE)
            ),
        )

    def _df_to_binding_predictions(
        self, df: pd.DataFrame, peptides: list[str]
    ) -> list[BindingPrediction]:
        """Convert standardised DataFrame to ``BindingPrediction`` list.

        Uses ``peptide_num`` for ordering when available; falls back to
        DataFrame row order (which matches MHCflurry's input order).
        """
        if COL_PEPTIDE_NUM in df.columns:
            df = df.sort_values(COL_PEPTIDE_NUM)
        return [
            self._row_to_binding_prediction(row, peptides[i])
            for i, (_, row) in enumerate(df.iterrows())
        ]

    def _df_to_prediction_results(
        self, df: pd.DataFrame, candidates: list[PeptideCandidate]
    ) -> list[MHCIPredictionResult]:
        """Convert standardised DataFrame to ``MHCIPredictionResult`` list."""
        if COL_PEPTIDE_NUM in df.columns:
            df = df.sort_values(COL_PEPTIDE_NUM)
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            c = candidates[i]
            results.append(
                MHCIPredictionResult(
                    peptide=str(row.get(COL_PEPTIDE, c.peptide_sequence)),
                    best_allele=str(row.get(COL_BEST_ALLELE, "")),
                    affinity_nm=float(row.get(COL_MHCFLURRY_AFFINITY, 0.0)),
                    affinity_percentile=float(
                        row.get(COL_AFFINITY_PERCENTILE, _DEFAULT_AFFINITY_PERCENTILE)
                    ),
                    processing_score=float(
                        row.get(COL_PROCESSING_SCORE, _DEFAULT_PROCESSING_SCORE)
                    ),
                    presentation_score=float(row.get(COL_PRESENTATION_SCORE, 0.0)),
                    presentation_percentile=float(
                        row.get(
                            COL_PRESENTATION_PERCENTILE,
                            _DEFAULT_PRESENTATION_PERCENTILE,
                        )
                    ),
                    mutation_str=c.mutation_str,
                    transcript_id=c.transcript_id,
                    gene=c.gene,
                    aa_pos=c.aa_pos,
                    n_flank=c.n_flank,
                    c_flank=c.c_flank,
                )
            )
        return results

    def _df_to_wildtype_results(
        self,
        df: pd.DataFrame,
        wt_peptides: list[str],
        mut_peptides: list[str],
    ) -> list[WildtypePredictionResult]:
        """Convert standardised DataFrame to ``WildtypePredictionResult`` list."""
        if COL_PEPTIDE_NUM in df.columns:
            df = df.sort_values(COL_PEPTIDE_NUM)
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append(
                WildtypePredictionResult(
                    wt_peptide=str(row.get(COL_PEPTIDE, wt_peptides[i])),
                    mut_peptide=mut_peptides[i],
                    wildtype_affinity_nm=float(row.get(COL_MHCFLURRY_AFFINITY, 0.0)),
                )
            )
        return results
