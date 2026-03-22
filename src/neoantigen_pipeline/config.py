"""Pipeline configuration dataclasses.

All pipeline parameters are defined here as frozen dataclasses and loaded
once at startup via dependency injection. No global configuration state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neoantigen_pipeline.exceptions import ConfigurationError


@dataclass(frozen=True)
class MHCIPredictionConfig:
    """Configuration for MHC class I binding prediction.

    Attributes:
        alleles: HLA alleles to predict for, e.g. ("HLA-A*02:01",).
        peptide_lengths: Peptide lengths to tile. Defaults to 8-11mers.
        use_presentation_score: Whether to use mhcflurry presentation score.
        binding_affinity_threshold_nm: IC50 threshold in nM for binders.
        percentile_rank_threshold: Percentile rank threshold for binders.
    """

    alleles: tuple[str, ...]
    peptide_lengths: tuple[int, ...] = (8, 9, 10, 11)
    use_presentation_score: bool = True
    binding_affinity_threshold_nm: float = 500.0
    percentile_rank_threshold: float = 2.0


@dataclass(frozen=True)
class PeptideGenerationConfig:
    """Configuration for peptide tiling from mutant protein sequences.

    Attributes:
        peptide_lengths: k-mer lengths to generate.
        n_flank_length: Length of N-terminal flanking sequence for processing.
        c_flank_length: Length of C-terminal flanking sequence for processing.
    """

    peptide_lengths: tuple[int, ...] = (8, 9, 10, 11)
    n_flank_length: int = 10
    c_flank_length: int = 10


@dataclass(frozen=True)
class ExpressionFilterConfig:
    """Configuration for expression-based variant filtering.

    Attributes:
        min_expression: Minimum expression level (TPM/FPKM) to retain a variant.
        expression_field: VCF INFO field containing expression data.
        filter_missing: Whether to filter out variants with no expression data.
    """

    min_expression: float = 1.0
    expression_field: str = "CSQ"
    filter_missing: bool = False


@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for composite neoantigen scoring.

    Weights must sum to 1.0 for a well-calibrated composite score.

    Attributes:
        presentation_score_weight: Weight for MHC presentation score.
        agretopicity_weight: Weight for agretopicity (mutant vs wildtype binding).
        expression_weight: Weight for gene expression level.
        vaf_weight: Weight for variant allele fraction.
    """

    presentation_score_weight: float = 0.4
    agretopicity_weight: float = 0.2
    expression_weight: float = 0.2
    vaf_weight: float = 0.2


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the neoantigen prediction pipeline.

    Attributes:
        mhc_i: MHC class I prediction settings.
        peptide_generation: Peptide tiling settings.
        expression_filter: Expression filtering settings.
        scoring: Composite scoring settings.
        output_dir: Directory for output files.
    """

    mhc_i: MHCIPredictionConfig
    peptide_generation: PeptideGenerationConfig
    expression_filter: ExpressionFilterConfig
    scoring: ScoringConfig
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        """Load pipeline configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated PipelineConfig instance.

        Raises:
            ConfigurationError: If the file cannot be read or contains
                invalid configuration.
        """
        try:
            config_text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot read config file '{path}': {exc}"
            ) from exc

        try:
            raw: dict[str, Any] = yaml.safe_load(config_text) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"YAML parse error in '{path}': {exc}") from exc

        try:
            mhc_i_raw = raw.get("mhc_i", {})
            mhc_i = MHCIPredictionConfig(
                alleles=tuple(mhc_i_raw.get("alleles", [])),
                peptide_lengths=tuple(mhc_i_raw.get("peptide_lengths", (8, 9, 10, 11))),
                use_presentation_score=mhc_i_raw.get("use_presentation_score", True),
                binding_affinity_threshold_nm=float(
                    mhc_i_raw.get("binding_affinity_threshold_nm", 500.0)
                ),
                percentile_rank_threshold=float(
                    mhc_i_raw.get("percentile_rank_threshold", 2.0)
                ),
            )

            pg_raw = raw.get("peptide_generation", {})
            peptide_generation = PeptideGenerationConfig(
                peptide_lengths=tuple(pg_raw.get("peptide_lengths", (8, 9, 10, 11))),
                n_flank_length=int(pg_raw.get("n_flank_length", 10)),
                c_flank_length=int(pg_raw.get("c_flank_length", 10)),
            )

            ef_raw = raw.get("expression_filter", {})
            expression_filter = ExpressionFilterConfig(
                min_expression=float(ef_raw.get("min_expression", 1.0)),
                expression_field=str(ef_raw.get("expression_field", "CSQ")),
                filter_missing=bool(ef_raw.get("filter_missing", False)),
            )

            sc_raw = raw.get("scoring", {})
            scoring = ScoringConfig(
                presentation_score_weight=float(
                    sc_raw.get("presentation_score_weight", 0.4)
                ),
                agretopicity_weight=float(sc_raw.get("agretopicity_weight", 0.2)),
                expression_weight=float(sc_raw.get("expression_weight", 0.2)),
                vaf_weight=float(sc_raw.get("vaf_weight", 0.2)),
            )

            return cls(
                mhc_i=mhc_i,
                peptide_generation=peptide_generation,
                expression_filter=expression_filter,
                scoring=scoring,
                output_dir=str(raw.get("output_dir", "results")),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigurationError(
                f"Invalid configuration in '{path}': {exc}"
            ) from exc
