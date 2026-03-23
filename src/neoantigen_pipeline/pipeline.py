"""End-to-end neoantigen prediction pipeline orchestrator.

Coordinates all pipeline steps from VCF input to a ranked
``NeoantigenResultSet``, with per-step timing and structured logging.
All inter-step data is passed as typed dataclasses — DataFrames are
confined to the prediction backend internals and the final result serialisation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from neoantigen_pipeline.candidates.expression_filter import ExpressionFilter
from neoantigen_pipeline.candidates.peptide_generator import PeptideGenerator
from neoantigen_pipeline.config import PipelineConfig
from neoantigen_pipeline.io.hla_reader import HLAReader
from neoantigen_pipeline.io.proteome import ProteomeDB
from neoantigen_pipeline.io.vcf_reader import VCFReader
from neoantigen_pipeline.prediction.mhcflurry import MHCflurryPredictor
from neoantigen_pipeline.prediction.results import ScoredCandidate
from neoantigen_pipeline.results.neoantigen import NeoantigenCandidate, NeoantigenResultSet
from neoantigen_pipeline.scoring.agretopicity import AgretopicityScorer
from neoantigen_pipeline.scoring.ranking import RankingScorer

if TYPE_CHECKING:
    from neoantigen_pipeline.candidates.peptide_generator import PeptideCandidate
    from neoantigen_pipeline.io.vcf_reader import SomaticVariant
    from neoantigen_pipeline.prediction.results import (
        MHCIPredictionResult,
        WildtypePredictionResult,
    )


class NeoantigenPipeline:
    """End-to-end neoantigen prediction pipeline.

    Orchestrates the full pipeline from annotated VCF input to ranked
    neoantigen candidates. Each step is extracted into a private ``_step_*``
    method for readability; ``run()`` is a clean sequential call chain.

    Steps:
    1. Load patient HLA types (OptiType TSV or config override).
    2. Read missense variants from a VEP-annotated VCF.
    3. Filter variants by gene expression.
    4. Load the reference proteome.
    5. Generate mutant and wildtype k-mer peptide candidates.
    6. Predict MHC-I binding and presentation (mutant + wildtype).
    7. Compute agretopicity; attach expression and VAF; assemble ScoredCandidates.
    8. Rank candidates by composite score.

    Args:
        config: Fully populated ``PipelineConfig`` instance.
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
            Configured ``NeoantigenPipeline`` instance.

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
            Ranked ``NeoantigenResultSet``.

        Raises:
            VCFParsingError: If the VCF cannot be read.
            HLAParsingError: If HLA types cannot be parsed.
            ProteomeError: If the proteome FASTA cannot be loaded.
            PredictionError: If MHC binding prediction fails.
        """
        t_start = time.perf_counter()
        self._logger.info("=== NeoantigenPipeline: starting run ===")
        self._logger.info("  VCF:      %s", vcf_path)
        self._logger.info("  HLA:      %s", hla_path)
        self._logger.info("  Proteome: %s", proteome_path)

        alleles = self._step_load_hla(hla_path)

        variants = self._step_read_variants(vcf_path)
        if not variants:
            self._logger.warning("No missense variants found; returning empty result set.")
            return NeoantigenResultSet([])

        variants = self._step_filter_expression(variants)
        if not variants:
            self._logger.warning("No variants remain after expression filtering.")
            return NeoantigenResultSet([])

        proteome_db = self._step_load_proteome(proteome_path)

        all_candidates = self._step_generate_peptides(variants, proteome_db)
        if not all_candidates:
            self._logger.warning("No peptide candidates generated.")
            return NeoantigenResultSet([])

        mutant_results, wildtype_results = self._step_predict_binding(all_candidates, alleles)
        if not mutant_results:
            self._logger.warning("MHC prediction returned no results.")
            return NeoantigenResultSet([])

        scored = self._step_score_candidates(mutant_results, wildtype_results, variants)
        ranked = self._step_rank(scored)

        self._logger.info(
            "=== Pipeline complete: %d candidates in %.2fs ===",
            len(ranked),
            time.perf_counter() - t_start,
        )
        return NeoantigenResultSet(ranked)

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
            self._N_STEPS, len(alleles), source, time.perf_counter() - t0,
        )
        return alleles

    def _step_read_variants(self, vcf_path: str) -> list[SomaticVariant]:
        t0 = time.perf_counter()
        self._logger.info("[Step 2/%d] Reading missense variants from VCF...", self._N_STEPS)
        variants = VCFReader(vcf_path).read_missense_variants()
        self._logger.info(
            "[Step 2/%d] Found %d missense variants. Done in %.2fs",
            self._N_STEPS, len(variants), time.perf_counter() - t0,
        )
        return variants

    def _step_filter_expression(self, variants: list[SomaticVariant]) -> list[SomaticVariant]:
        t0 = time.perf_counter()
        self._logger.info("[Step 3/%d] Filtering variants by expression...", self._N_STEPS)
        variants = ExpressionFilter(self._config.expression_filter).filter(variants)
        self._logger.info(
            "[Step 3/%d] %d variants after expression filter. Done in %.2fs",
            self._N_STEPS, len(variants), time.perf_counter() - t0,
        )
        return variants

    def _step_load_proteome(self, proteome_path: str) -> ProteomeDB:
        t0 = time.perf_counter()
        self._logger.info("[Step 4/%d] Loading reference proteome...", self._N_STEPS)
        proteome_db = ProteomeDB(proteome_path)
        self._logger.info(
            "[Step 4/%d] Loaded %d sequences. Done in %.2fs",
            self._N_STEPS, proteome_db.size, time.perf_counter() - t0,
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
            self._N_STEPS, len(all_candidates), len(variants), time.perf_counter() - t0,
        )
        return all_candidates

    def _step_predict_binding(
        self, candidates: list[PeptideCandidate], alleles: list[str]
    ) -> tuple[list[MHCIPredictionResult], list[WildtypePredictionResult]]:
        t0 = time.perf_counter()
        self._logger.info("[Step 6/%d] Running MHC-I predictions...", self._N_STEPS)
        predictor = MHCflurryPredictor(self._config.mhc_i)
        mutant_results = predictor.predict_with_processing(candidates, alleles)
        wildtype_results = predictor.predict_wildtype(candidates, alleles)
        self._logger.info(
            "[Step 6/%d] Prediction complete (%d mutant, %d wildtype). Done in %.2fs",
            self._N_STEPS, len(mutant_results), len(wildtype_results), time.perf_counter() - t0,
        )
        return mutant_results, wildtype_results

    def _step_score_candidates(
        self,
        mutant_results: list[MHCIPredictionResult],
        wildtype_results: list[WildtypePredictionResult],
        variants: list[SomaticVariant],
    ) -> list[ScoredCandidate]:
        t0 = time.perf_counter()
        self._logger.info("[Step 7/%d] Computing agretopicity scores...", self._N_STEPS)

        variant_meta: dict[str, tuple[float, float]] = {
            f"{v.gene}_{v.protein_change}": (
                v.expression if v.expression is not None else 0.0,
                v.vaf,
            )
            for v in variants
        }

        agretopicities = AgretopicityScorer().compute_batch(
            [m.affinity_nm for m in mutant_results],
            [w.wildtype_affinity_nm for w in wildtype_results],
        )

        scored: list[ScoredCandidate] = [
            ScoredCandidate(
                peptide=mut.peptide,
                wildtype_peptide=wt.wt_peptide,
                gene=mut.gene,
                mutation_str=mut.mutation_str,
                best_allele=mut.best_allele,
                presentation_score=mut.presentation_score,
                binding_affinity_nm=mut.affinity_nm,
                wildtype_affinity_nm=wt.wildtype_affinity_nm,
                processing_score=mut.processing_score,
                agretopicity=agretopicities[i],
                expression=variant_meta.get(mut.mutation_str, (0.0, 0.0))[0],
                vaf=variant_meta.get(mut.mutation_str, (0.0, 0.0))[1],
            )
            for i, (mut, wt) in enumerate(zip(mutant_results, wildtype_results))
        ]

        self._logger.info(
            "[Step 7/%d] Assembled %d scored candidates. Done in %.2fs",
            self._N_STEPS, len(scored), time.perf_counter() - t0,
        )
        return scored

    def _step_rank(self, candidates: list[ScoredCandidate]) -> list[NeoantigenCandidate]:
        t0 = time.perf_counter()
        self._logger.info("[Step 8/%d] Ranking candidates...", self._N_STEPS)
        ranked = RankingScorer(self._config.scoring).rank(candidates)
        self._logger.info(
            "[Step 8/%d] Ranked %d candidates. Done in %.2fs",
            self._N_STEPS, len(ranked), time.perf_counter() - t0,
        )
        return ranked
