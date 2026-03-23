"""End-to-end neoantigen prediction pipeline orchestrator.

Coordinates all pipeline steps from VCF input to ranked NeoantigenResultSet
output, with per-step timing and structured logging.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd

from neoantigen_pipeline._constants import (
    COL_AGRETOPICITY,
    COL_BEST_ALLELE,
    COL_COMPOSITE_RANK,
    COL_COMPOSITE_SCORE,
    COL_EXPRESSION,
    COL_GENE,
    COL_MHCFLURRY_AFFINITY,
    COL_MUTATION_STR,
    COL_MUT_PEPTIDE,
    COL_PEPTIDE,
    COL_PROCESSING_SCORE,
    COL_PRESENTATION_SCORE,
    COL_VAF,
    COL_WILDTYPE_AFFINITY,
)
from neoantigen_pipeline.config import PipelineConfig
from neoantigen_pipeline.exceptions import PredictionError
from neoantigen_pipeline.io.hla_reader import HLAReader
from neoantigen_pipeline.io.proteome import ProteomeDB
from neoantigen_pipeline.io.vcf_reader import VCFReader
from neoantigen_pipeline.prediction.mhc_i import MHCIPredictor
from neoantigen_pipeline.processing.expression_filter import ExpressionFilter
from neoantigen_pipeline.processing.peptide_generator import PeptideGenerator
from neoantigen_pipeline.results.neoantigen import (
    NeoantigenCandidate,
    NeoantigenResultSet,
)
from neoantigen_pipeline.scoring.agretopicity import AgretopicityScorer
from neoantigen_pipeline.scoring.ranking import RankingScorer

if TYPE_CHECKING:
    from neoantigen_pipeline.io.vcf_reader import SomaticVariant
    from neoantigen_pipeline.processing.peptide_generator import PeptideCandidate


class NeoantigenPipeline:
    """End-to-end neoantigen prediction pipeline.

    Orchestrates the full pipeline from annotated VCF input to ranked
    neoantigen candidates. Each pipeline step is timed and logged.

    The pipeline performs the following steps:
    1. Load patient HLA types (from OptiType TSV or config).
    2. Read missense variants from a VEP-annotated VCF.
    3. Filter variants by gene expression.
    4. Load the reference proteome.
    5. Generate mutant and wildtype k-mer peptide candidates.
    6. Predict MHC-I binding and presentation scores (mutant + wildtype).
    7. Compute agretopicity scores.
    8. Rank candidates by composite score.
    9. Return a NeoantigenResultSet.

    Args:
        config: Fully populated PipelineConfig instance.
    """

    _N_STEPS: int = 8

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(type(self).__qualname__)

    @classmethod
    def from_yaml(cls, config_path: str) -> NeoantigenPipeline:
        """Construct a pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML config file.

        Returns:
            Configured NeoantigenPipeline instance.

        Raises:
            ConfigurationError: If the config file cannot be read or parsed.
        """
        return cls(PipelineConfig.from_yaml(config_path))

    def run(
        self,
        vcf_path: str,
        hla_path: str,
        proteome_path: str,
    ) -> NeoantigenResultSet:
        """Execute the full neoantigen prediction pipeline.

        Args:
            vcf_path: Path to VEP-annotated somatic variant VCF (gzipped ok).
            hla_path: Path to OptiType TSV result file.
            proteome_path: Path to Ensembl peptide FASTA (gzipped ok).

        Returns:
            Ranked NeoantigenResultSet.

        Raises:
            VCFParsingError: If the VCF cannot be read.
            HLAParsingError: If HLA types cannot be parsed.
            ProteomeError: If the proteome FASTA cannot be loaded.
            PredictionError: If MHC binding prediction fails.
        """
        pipeline_start = time.perf_counter()
        self._logger.info("=== NeoantigenPipeline: starting run ===")
        self._logger.info("  VCF:      %s", vcf_path)
        self._logger.info("  HLA:      %s", hla_path)
        self._logger.info("  Proteome: %s", proteome_path)

        # ------------------------------------------------------------------
        # Step 1: Load HLA types
        # ------------------------------------------------------------------
        alleles = self._step_load_hla(hla_path)

        # ------------------------------------------------------------------
        # Step 2: Read missense variants from VCF
        # ------------------------------------------------------------------
        variants = self._step_read_variants(vcf_path)
        if not variants:
            self._logger.warning("No missense variants found; returning empty result set.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 3: Filter by expression
        # ------------------------------------------------------------------
        variants = self._step_filter_expression(variants)
        if not variants:
            self._logger.warning("No variants remain after expression filtering.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 4: Load proteome
        # ------------------------------------------------------------------
        proteome_db = self._step_load_proteome(proteome_path)

        # ------------------------------------------------------------------
        # Step 5: Generate peptide candidates
        # ------------------------------------------------------------------
        all_candidates = self._step_generate_peptides(variants, proteome_db)
        if not all_candidates:
            self._logger.warning("No peptide candidates generated.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 6: MHC-I prediction (mutant + wildtype)
        # ------------------------------------------------------------------
        mutant_df, wildtype_df = self._step_predict_binding(all_candidates, alleles)
        if mutant_df.empty:
            self._logger.warning("MHC prediction returned empty results.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 7: Compute agretopicity
        # ------------------------------------------------------------------
        combined_df = self._step_compute_agretopicity(mutant_df, wildtype_df, variants)

        # ------------------------------------------------------------------
        # Step 8: Ranking
        # ------------------------------------------------------------------
        ranked_df = self._step_rank(combined_df)

        result_set = self._build_result_set(ranked_df)
        self._logger.info(
            "=== Pipeline complete: %d candidates in %.2fs ===",
            len(result_set),
            time.perf_counter() - pipeline_start,
        )
        return result_set

    # ── Private step methods ─────────────────────────────────────────────────

    def _step_load_hla(self, hla_path: str) -> list[str]:
        t0 = time.perf_counter()
        self._logger.info("[Step 1/%d] Loading HLA types...", self._N_STEPS)
        patient_hla = HLAReader().read_optitype(hla_path)
        alleles = (
            list(self._config.mhc_i.alleles)
            if self._config.mhc_i.alleles
            else list(patient_hla.class_i_alleles)
        )
        source = "config" if self._config.mhc_i.alleles else "HLA file"
        self._logger.info(
            "[Step 1/%d] Using %d alleles from %s. Done in %.2fs",
            self._N_STEPS,
            len(alleles),
            source,
            time.perf_counter() - t0,
        )
        return alleles

    def _step_read_variants(self, vcf_path: str) -> list[SomaticVariant]:
        t0 = time.perf_counter()
        self._logger.info("[Step 2/%d] Reading missense variants from VCF...", self._N_STEPS)
        variants = VCFReader(vcf_path).read_missense_variants()
        self._logger.info(
            "[Step 2/%d] Found %d missense variants. Done in %.2fs",
            self._N_STEPS,
            len(variants),
            time.perf_counter() - t0,
        )
        return variants

    def _step_filter_expression(self, variants: list[SomaticVariant]) -> list[SomaticVariant]:
        t0 = time.perf_counter()
        self._logger.info("[Step 3/%d] Filtering variants by expression...", self._N_STEPS)
        variants = ExpressionFilter(self._config.expression_filter).filter(variants)
        self._logger.info(
            "[Step 3/%d] %d variants after expression filter. Done in %.2fs",
            self._N_STEPS,
            len(variants),
            time.perf_counter() - t0,
        )
        return variants

    def _step_load_proteome(self, proteome_path: str) -> ProteomeDB:
        t0 = time.perf_counter()
        self._logger.info("[Step 4/%d] Loading reference proteome...", self._N_STEPS)
        proteome_db = ProteomeDB(proteome_path)
        self._logger.info(
            "[Step 4/%d] Loaded %d sequences. Done in %.2fs",
            self._N_STEPS,
            proteome_db.size,
            time.perf_counter() - t0,
        )
        return proteome_db

    def _step_generate_peptides(
        self, variants: list[SomaticVariant], proteome_db: ProteomeDB
    ) -> list[PeptideCandidate]:
        t0 = time.perf_counter()
        self._logger.info("[Step 5/%d] Generating peptide candidates...", self._N_STEPS)
        generator = PeptideGenerator(self._config.peptide_generation, proteome_db)
        all_candidates: list[PeptideCandidate] = []
        for variant in variants:
            all_candidates.extend(generator.generate(variant))
        self._logger.info(
            "[Step 5/%d] Generated %d candidates from %d variants. Done in %.2fs",
            self._N_STEPS,
            len(all_candidates),
            len(variants),
            time.perf_counter() - t0,
        )
        return all_candidates

    def _step_predict_binding(
        self, candidates: list[PeptideCandidate], alleles: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        t0 = time.perf_counter()
        self._logger.info("[Step 6/%d] Running MHC-I predictions...", self._N_STEPS)
        predictor = MHCIPredictor(self._config.mhc_i)
        try:
            mutant_df = predictor.predict_with_processing(candidates, alleles)
            wildtype_df = predictor.predict_wildtype(candidates, alleles)
        except PredictionError:
            raise
        self._logger.info(
            "[Step 6/%d] Prediction complete (%d mutant, %d wildtype rows). Done in %.2fs",
            self._N_STEPS,
            len(mutant_df),
            len(wildtype_df),
            time.perf_counter() - t0,
        )
        return mutant_df, wildtype_df

    def _step_compute_agretopicity(
        self,
        mutant_df: pd.DataFrame,
        wildtype_df: pd.DataFrame,
        variants: list[SomaticVariant],
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        self._logger.info("[Step 7/%d] Computing agretopicity scores...", self._N_STEPS)
        combined_df = self._merge_wildtype_affinities(mutant_df, wildtype_df)
        if COL_WILDTYPE_AFFINITY in combined_df.columns:
            combined_df = AgretopicityScorer().annotate_dataframe(combined_df)
        else:
            combined_df[COL_AGRETOPICITY] = 1.0
            self._logger.warning(
                "'%s' column absent; agretopicity set to 1.0", COL_WILDTYPE_AFFINITY
            )
        combined_df = self._attach_variant_metadata(combined_df, variants)
        self._logger.info(
            "[Step 7/%d] Agretopicity computed. Done in %.2fs",
            self._N_STEPS,
            time.perf_counter() - t0,
        )
        return combined_df

    def _step_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.perf_counter()
        self._logger.info("[Step 8/%d] Ranking candidates...", self._N_STEPS)
        ranked_df = RankingScorer(self._config.scoring).rank(df)
        self._logger.info(
            "[Step 8/%d] Ranked %d candidates. Done in %.2fs",
            self._N_STEPS,
            len(ranked_df),
            time.perf_counter() - t0,
        )
        return ranked_df

    # ── Helper methods ───────────────────────────────────────────────────────

    def _merge_wildtype_affinities(
        self, mutant_df: pd.DataFrame, wildtype_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge wildtype affinity values into the mutant prediction DataFrame.

        Joins on (mutant peptide sequence, best allele): ``mutant_df[COL_PEPTIDE]``
        is matched against ``wildtype_df[COL_MUT_PEPTIDE]``, which was added by
        ``MHCIPredictor.predict_wildtype`` to preserve the mutant→wildtype mapping.

        Args:
            mutant_df: Mutant prediction DataFrame.
            wildtype_df: Wildtype prediction DataFrame with ``mut_peptide`` column.

        Returns:
            Mutant DataFrame with a ``wildtype_affinity`` column added.
        """
        if wildtype_df.empty or COL_WILDTYPE_AFFINITY not in wildtype_df.columns:
            self._logger.warning(
                "Wildtype prediction DataFrame is empty or missing '%s'",
                COL_WILDTYPE_AFFINITY,
            )
            result = mutant_df.copy()
            result[COL_WILDTYPE_AFFINITY] = float("nan")
            return result

        if COL_MUT_PEPTIDE not in wildtype_df.columns:
            self._logger.warning(
                "Wildtype DataFrame missing '%s'; falling back to positional merge",
                COL_MUT_PEPTIDE,
            )
            result = mutant_df.copy()
            if len(wildtype_df) == len(mutant_df):
                result[COL_WILDTYPE_AFFINITY] = wildtype_df[COL_WILDTYPE_AFFINITY].values
            else:
                result[COL_WILDTYPE_AFFINITY] = float("nan")
            return result

        merged = mutant_df.merge(
            wildtype_df[[COL_MUT_PEPTIDE, COL_BEST_ALLELE, COL_WILDTYPE_AFFINITY]],
            left_on=[COL_PEPTIDE, COL_BEST_ALLELE],
            right_on=[COL_MUT_PEPTIDE, COL_BEST_ALLELE],
            how="left",
        ).drop(columns=[COL_MUT_PEPTIDE])

        return merged

    def _attach_variant_metadata(
        self, df: pd.DataFrame, variants: list[SomaticVariant]
    ) -> pd.DataFrame:
        """Attach expression and VAF from the original variant list to the DataFrame.

        Matches on ``mutation_str`` (gene + protein_change).

        Args:
            df: Prediction DataFrame.
            variants: Original variant list.

        Returns:
            DataFrame with ``expression`` and ``vaf`` columns attached.
        """
        variant_meta: dict[str, tuple[float, float]] = {
            f"{v.gene}_{v.protein_change}": (
                v.expression if v.expression is not None else 0.0,
                v.vaf,
            )
            for v in variants
        }

        df = df.copy()
        if COL_MUTATION_STR in df.columns:
            df[COL_EXPRESSION] = df[COL_MUTATION_STR].map(
                lambda m: variant_meta.get(m, (0.0, 0.0))[0]
            )
            df[COL_VAF] = df[COL_MUTATION_STR].map(
                lambda m: variant_meta.get(m, (0.0, 0.0))[1]
            )
        else:
            df[COL_EXPRESSION] = 0.0
            df[COL_VAF] = 0.0
            self._logger.warning(
                "'%s' column absent; expression and VAF set to 0.0", COL_MUTATION_STR
            )
        return df

    def _build_result_set(self, ranked_df: pd.DataFrame) -> NeoantigenResultSet:
        """Convert the ranked DataFrame to a NeoantigenResultSet.

        Args:
            ranked_df: Fully scored and ranked prediction DataFrame.

        Returns:
            NeoantigenResultSet with one candidate per row.
        """

        def _safe_float(val: object, default: float = 0.0) -> float:
            try:
                f = float(val)  # type: ignore[arg-type]
                return f if pd.notna(f) else default
            except (TypeError, ValueError):
                return default

        def _safe_str(val: object, default: str = "") -> str:
            return str(val) if pd.notna(val) else default

        candidates = [
            NeoantigenCandidate(
                gene=_safe_str(row.get(COL_GENE, "")),
                mutation=_safe_str(row.get(COL_MUTATION_STR, "")),
                peptide=_safe_str(row.get(COL_PEPTIDE, "")),
                wildtype_peptide=_safe_str(row.get("wildtype_sequence", "")),
                best_allele=_safe_str(row.get(COL_BEST_ALLELE, "")),
                presentation_score=_safe_float(row.get(COL_PRESENTATION_SCORE)),
                binding_affinity_nm=_safe_float(row.get(COL_MHCFLURRY_AFFINITY)),
                wildtype_affinity_nm=_safe_float(row.get(COL_WILDTYPE_AFFINITY)),
                processing_score=_safe_float(row.get(COL_PROCESSING_SCORE)),
                agretopicity=_safe_float(row.get(COL_AGRETOPICITY, 1.0)),
                expression=_safe_float(row.get(COL_EXPRESSION)),
                vaf=_safe_float(row.get(COL_VAF)),
                composite_score=_safe_float(row.get(COL_COMPOSITE_SCORE)),
                composite_rank=int(row.get(COL_COMPOSITE_RANK, 0)),
            )
            for _, row in ranked_df.iterrows()
        ]
        return NeoantigenResultSet(candidates)
