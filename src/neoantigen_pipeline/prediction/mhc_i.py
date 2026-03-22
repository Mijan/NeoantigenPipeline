"""MHC class I binding prediction using MHCflurry.

Wraps the MHCflurry Class1PresentationPredictor to provide mutant and
wildtype peptide binding predictions with processing score support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from neoantigen_pipeline.exceptions import PredictionError
from neoantigen_pipeline.prediction.base import BindingPredictor

if TYPE_CHECKING:
    from neoantigen_pipeline.config import MHCIPredictionConfig
    from neoantigen_pipeline.processing.peptide_generator import PeptideCandidate


class MHCIPredictor(BindingPredictor):
    """MHC class I binding and presentation predictor backed by MHCflurry.

    Uses the MHCflurry ``Class1PresentationPredictor`` which predicts
    MHC-I binding affinity, antigen processing, and a combined presentation
    score for each peptide-allele pair.

    The predictor is loaded lazily on first use to avoid import overhead
    when the class is instantiated.

    Args:
        config: MHC-I prediction configuration specifying alleles, lengths,
            score thresholds, and whether to use presentation scores.

    Raises:
        PredictionError: If the MHCflurry models cannot be loaded.
    """

    def __init__(self, config: MHCIPredictionConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(type(self).__qualname__)
        self._predictor = None  # Lazy-loaded

    @property
    def name(self) -> str:
        """Predictor name.

        Returns:
            "MHCflurry2-Class1Presentation"
        """
        return "MHCflurry2-Class1Presentation"

    @property
    def mhc_class(self) -> int:
        """MHC class.

        Returns:
            1 (class I)
        """
        return 1

    def _load_predictor(self) -> Any:
        """Load and cache the MHCflurry Class1PresentationPredictor.

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
            raise PredictionError(
                "mhcflurry is not installed. "
                "Install with: pip install mhcflurry && mhcflurry-downloads fetch"
            ) from exc
        except Exception as exc:
            raise PredictionError(f"Failed to load MHCflurry models: {exc}") from exc

        return self._predictor

    def predict(self, peptides: list[str], alleles: list[str]) -> pd.DataFrame:
        """Predict MHC-I binding for peptides and alleles without flank context.

        Args:
            peptides: List of amino acid sequences.
            alleles: List of HLA allele strings.

        Returns:
            DataFrame with columns: peptide, allele, mhcflurry_affinity,
            mhcflurry_affinity_percentile, mhcflurry_presentation_score.

        Raises:
            PredictionError: If prediction fails.
        """
        return self._run_prediction(
            peptides=peptides,
            alleles=alleles,
            n_flanks=None,
            c_flanks=None,
        )

    def predict_with_processing(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> pd.DataFrame:
        """Predict MHC-I presentation for peptide candidates with flank context.

        Passes N- and C-terminal flanking sequences to MHCflurry's antigen
        processing model for improved prediction accuracy.

        Args:
            candidates: List of PeptideCandidate objects containing sequences
                and flanking regions.
            alleles: List of HLA allele strings to predict against.

        Returns:
            DataFrame with columns: peptide, allele, mhcflurry_affinity,
            mhcflurry_affinity_percentile, mhcflurry_processing_score,
            mhcflurry_presentation_score, mutation_str, transcript_id,
            n_flank, c_flank.

        Raises:
            PredictionError: If prediction fails.
        """
        if not candidates:
            return pd.DataFrame()

        peptides = [c.peptide_sequence for c in candidates]
        n_flanks = [c.n_flank for c in candidates]
        c_flanks = [c.c_flank for c in candidates]

        df = self._run_prediction(
            peptides=peptides,
            alleles=alleles,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
        )

        # Attach candidate metadata by merging on peptide sequence
        meta = pd.DataFrame(
            {
                "peptide": peptides,
                "mutation_str": [c.mutation_str for c in candidates],
                "transcript_id": [c.transcript_id for c in candidates],
                "gene": [c.gene for c in candidates],
                "aa_pos": [c.aa_pos for c in candidates],
                "n_flank": n_flanks,
                "c_flank": c_flanks,
            }
        ).drop_duplicates(subset=["peptide"])

        df = df.merge(meta, on="peptide", how="left")
        return df

    def predict_wildtype(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> pd.DataFrame:
        """Predict MHC-I binding for wildtype counterparts of mutant peptides.

        Used to compute agretopicity (ratio of wildtype to mutant binding).

        Args:
            candidates: List of PeptideCandidate objects; the wildtype_sequence
                field is used as the query peptide.
            alleles: List of HLA allele strings.

        Returns:
            DataFrame with columns: peptide (wildtype), allele, and prediction
            score columns. The "peptide" column contains wildtype sequences.

        Raises:
            PredictionError: If prediction fails.
        """
        if not candidates:
            return pd.DataFrame()

        wt_peptides = [c.wildtype_sequence for c in candidates]
        n_flanks = [c.n_flank for c in candidates]
        c_flanks = [c.c_flank for c in candidates]

        df = self._run_prediction(
            peptides=wt_peptides,
            alleles=alleles,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
        )
        # Rename affinity column to distinguish wildtype predictions
        if "mhcflurry_affinity" in df.columns:
            df = df.rename(columns={"mhcflurry_affinity": "wildtype_affinity"})

        return df

    def _run_prediction(
        self,
        peptides: list[str],
        alleles: list[str],
        n_flanks: list[str] | None,
        c_flanks: list[str] | None,
    ) -> pd.DataFrame:
        """Run MHCflurry prediction and standardise the output DataFrame.

        Args:
            peptides: Peptide sequences (one per candidate).
            alleles: HLA alleles to predict against.
            n_flanks: Optional N-terminal flanking sequences.
            c_flanks: Optional C-terminal flanking sequences.

        Returns:
            Standardised prediction DataFrame.

        Raises:
            PredictionError: If MHCflurry raises an error.
        """
        predictor = self._load_predictor()

        try:
            kwargs: dict = {}
            if n_flanks is not None:
                kwargs["n_flanks"] = n_flanks
            if c_flanks is not None:
                kwargs["c_flanks"] = c_flanks

            result = predictor.predict(
                peptides=peptides,
                alleles=alleles,
                **kwargs,
            )
        except Exception as exc:
            raise PredictionError(f"MHCflurry prediction failed: {exc}") from exc

        return self._standardise_columns(result)

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename MHCflurry output columns to pipeline-standard names.

        MHCflurry uses its own column naming scheme; this method maps those
        to the names expected by downstream scoring modules.

        Args:
            df: Raw MHCflurry prediction DataFrame.

        Returns:
            DataFrame with standardised column names.
        """
        rename_map = {
            "affinity": "mhcflurry_affinity",
            "affinity_percentile": "mhcflurry_affinity_percentile",
            "processing_score": "mhcflurry_processing_score",
            "presentation_score": "mhcflurry_presentation_score",
        }
        # Only rename columns that exist
        actual_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=actual_renames)

        return df
