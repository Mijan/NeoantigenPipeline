"""Pipeline configuration dataclasses.

All pipeline parameters are defined here as frozen dataclasses and loaded
once at startup via dependency injection. No global configuration state.
Default values are imported from ``_constants`` to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neoantigen_pipeline._constants import DEFAULT_FLANK_LENGTH, DEFAULT_PEPTIDE_LENGTHS
from neoantigen_pipeline.exceptions import ConfigurationError


@dataclass(frozen=True)
class MHCIPredictionConfig:
    """Configuration for MHC class I binding prediction.

    Attributes:
        alleles: HLA alleles to predict for, e.g. ("HLA-A*02:01",).
        peptide_lengths: Peptide lengths to tile. Defaults to 8–11mers.
        use_presentation_score: Whether to use mhcflurry presentation score.
        binding_affinity_threshold_nm: IC50 threshold in nM for binders.
        percentile_rank_threshold: Percentile rank threshold for binders.
    """

    alleles: tuple[str, ...]
    peptide_lengths: tuple[int, ...] = DEFAULT_PEPTIDE_LENGTHS
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

    peptide_lengths: tuple[int, ...] = DEFAULT_PEPTIDE_LENGTHS
    n_flank_length: int = DEFAULT_FLANK_LENGTH
    c_flank_length: int = DEFAULT_FLANK_LENGTH


@dataclass(frozen=True)
class ExpressionFilterConfig:
    """Configuration for expression-based variant filtering.

    Attributes:
        min_expression: Minimum expression level (TPM/FPKM) to retain a variant.
        filter_missing: Whether to filter out variants with no expression data.
    """

    min_expression: float = 1.0
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
class HLApolloConfig:
    """Configuration for HLApollo MHC-I prediction.

    Attributes:
        binary_path: Path to the HLA-Apollo executable.
        docker_image: If set, run via Docker instead of native binary.
        timeout_seconds: Max seconds to wait for the binary to finish.
        batch_size: Number of peptide-allele pairs per subprocess call.
        enabled: Whether to run HLApollo predictions (opt-in).
    """

    binary_path: str = "tools/HLApollo/HLA-Apollo"
    docker_image: str | None = "hla-apollo"
    timeout_seconds: int = 3600
    batch_size: int = 5000
    enabled: bool = False


@dataclass(frozen=True)
class ESMConfig:
    """Configuration for ESM-2 protein language model embeddings.

    Attributes:
        model_name: ESM-2 model identifier.
        cache_path: Path to the HDF5 embedding cache file.
        device: PyTorch device ("auto", "cuda", or "cpu").
        batch_size: Number of proteins to process per batch.
        context_window: Residues on each side of the peptide for context.
        enabled: Whether to compute ESM-2 embeddings (opt-in).
    """

    model_name: str = "esm2_t33_650M_UR50D"
    cache_path: str = "results/esm_cache.h5"
    device: str = "auto"
    batch_size: int = 4
    context_window: int = 15
    enabled: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the neoantigen prediction pipeline.

    Attributes:
        mhc_i: MHC class I prediction settings.
        peptide_generation: Peptide tiling settings.
        expression_filter: Expression filtering settings.
        scoring: Composite scoring settings.
        hlapollo: HLApollo predictor settings (optional).
        esm: ESM-2 embedding settings (optional).
        output_dir: Directory for output files.
    """

    mhc_i: MHCIPredictionConfig
    peptide_generation: PeptideGenerationConfig
    expression_filter: ExpressionFilterConfig
    scoring: ScoringConfig
    hlapollo: HLApolloConfig = HLApolloConfig()
    esm: ESMConfig = ESMConfig()
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
                peptide_lengths=tuple(
                    mhc_i_raw.get("peptide_lengths", DEFAULT_PEPTIDE_LENGTHS)
                ),
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
                peptide_lengths=tuple(
                    pg_raw.get("peptide_lengths", DEFAULT_PEPTIDE_LENGTHS)
                ),
                n_flank_length=int(pg_raw.get("n_flank_length", DEFAULT_FLANK_LENGTH)),
                c_flank_length=int(pg_raw.get("c_flank_length", DEFAULT_FLANK_LENGTH)),
            )

            ef_raw = raw.get("expression_filter", {})
            expression_filter = ExpressionFilterConfig(
                min_expression=float(ef_raw.get("min_expression", 1.0)),
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

            ha_raw = raw.get("hlapollo", {})
            hlapollo = HLApolloConfig(
                binary_path=str(ha_raw.get("binary_path", "tools/HLApollo/HLA-Apollo")),
                docker_image=ha_raw.get("docker_image") or None,
                timeout_seconds=int(ha_raw.get("timeout_seconds", 600)),
                batch_size=int(ha_raw.get("batch_size", 5000)),
                enabled=bool(ha_raw.get("enabled", False)),
            )

            esm_raw = raw.get("esm", {})
            esm = ESMConfig(
                model_name=str(esm_raw.get("model_name", "esm2_t33_650M_UR50D")),
                cache_path=str(esm_raw.get("cache_path", "results/esm_cache.h5")),
                device=str(esm_raw.get("device", "auto")),
                batch_size=int(esm_raw.get("batch_size", 4)),
                context_window=int(esm_raw.get("context_window", 15)),
                enabled=bool(esm_raw.get("enabled", False)),
            )

            return cls(
                mhc_i=mhc_i,
                peptide_generation=peptide_generation,
                expression_filter=expression_filter,
                scoring=scoring,
                hlapollo=hlapollo,
                esm=esm,
                output_dir=str(raw.get("output_dir", "results")),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ConfigurationError(
                f"Invalid configuration in '{path}': {exc}"
            ) from exc
