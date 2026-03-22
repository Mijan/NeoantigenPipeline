"""End-to-end neoantigen prediction pipeline orchestrator.

Coordinates all pipeline steps from VCF input to ranked NeoantigenResultSet
output, with per-step timing and structured logging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from neoantigen_pipeline.config import PipelineConfig
from neoantigen_pipeline.exceptions import PredictionError
from neoantigen_pipeline.io.hla_reader import HLAReader
from neoantigen_pipeline.io.proteome import ProteomeDB
from neoantigen_pipeline.io.vcf_reader import VCFReader
from neoantigen_pipeline.prediction.mhc_i import MHCIPredictor
from neoantigen_pipeline.processing.expression_filter import ExpressionFilter
from neoantigen_pipeline.processing.peptide_generator import PeptideGenerator
from neoantigen_pipeline.results.neoantigen import NeoantigenCandidate, NeoantigenResultSet
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
        config = PipelineConfig.from_yaml(config_path)
        return cls(config)

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
        t0 = time.perf_counter()
        self._logger.info("[Step 1/8] Loading HLA types...")
        hla_reader = HLAReader()
        patient_hla = hla_reader.read_optitype(hla_path)
        alleles = list(patient_hla.class_i_alleles)
        # If config specifies alleles, use those (overrides HLA file)
        if self._config.mhc_i.alleles:
            alleles = list(self._config.mhc_i.alleles)
            self._logger.info(
                "Using alleles from config (%d): %s", len(alleles), alleles
            )
        else:
            self._logger.info(
                "Using alleles from HLA file (%d): %s", len(alleles), alleles
            )
        self._logger.info("[Step 1/8] Done in %.2fs", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Step 2: Read missense variants from VCF
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 2/8] Reading missense variants from VCF...")
        vcf_reader = VCFReader(vcf_path)
        variants = vcf_reader.read_missense_variants()
        self._logger.info(
            "[Step 2/8] Found %d missense variants. Done in %.2fs",
            len(variants), time.perf_counter() - t0,
        )

        if not variants:
            self._logger.warning("No missense variants found; returning empty result set.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 3: Filter by expression
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 3/8] Filtering variants by expression...")
        expr_filter = ExpressionFilter(self._config.expression_filter)
        variants = expr_filter.filter(variants)
        self._logger.info(
            "[Step 3/8] %d variants after expression filter. Done in %.2fs",
            len(variants), time.perf_counter() - t0,
        )

        if not variants:
            self._logger.warning("No variants remain after expression filtering.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 4: Load proteome
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 4/8] Loading reference proteome...")
        proteome_db = ProteomeDB(proteome_path)
        self._logger.info(
            "[Step 4/8] Loaded %d sequences. Done in %.2fs",
            proteome_db.size, time.perf_counter() - t0,
        )

        # ------------------------------------------------------------------
        # Step 5: Generate peptide candidates
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 5/8] Generating peptide candidates...")
        peptide_gen = PeptideGenerator(self._config.peptide_generation, proteome_db)
        all_candidates: list[PeptideCandidate] = []
        for variant in variants:
            candidates = peptide_gen.generate(variant)
            all_candidates.extend(candidates)
        self._logger.info(
            "[Step 5/8] Generated %d peptide candidates from %d variants. Done in %.2fs",
            len(all_candidates), len(variants), time.perf_counter() - t0,
        )

        if not all_candidates:
            self._logger.warning("No peptide candidates generated.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 6: MHC-I prediction (mutant + wildtype)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 6/8] Running MHC-I predictions...")
        mhc_predictor = MHCIPredictor(self._config.mhc_i)

        try:
            mutant_df = mhc_predictor.predict_with_processing(all_candidates, alleles)
            wildtype_df = mhc_predictor.predict_wildtype(all_candidates, alleles)
        except PredictionError:
            raise

        self._logger.info(
            "[Step 6/8] Prediction complete (%d mutant, %d wildtype rows). Done in %.2fs",
            len(mutant_df), len(wildtype_df), time.perf_counter() - t0,
        )

        if mutant_df.empty:
            self._logger.warning("MHC prediction returned empty results.")
            return NeoantigenResultSet([])

        # ------------------------------------------------------------------
        # Step 7: Compute agretopicity
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 7/8] Computing agretopicity scores...")
        combined_df = self._merge_wildtype_affinities(mutant_df, wildtype_df)
        agretopicity_scorer = AgretopicityScorer()
        if "wildtype_affinity" in combined_df.columns:
            combined_df = agretopicity_scorer.annotate_dataframe(combined_df)
        else:
            combined_df["agretopicity"] = 1.0
            self._logger.warning("wildtype_affinity column absent; agretopicity set to 1.0")

        self._logger.info(
            "[Step 7/8] Agretopicity computed. Done in %.2fs",
            time.perf_counter() - t0,
        )

        # Attach expression and VAF from variant metadata
        combined_df = self._attach_variant_metadata(combined_df, variants)

        # ------------------------------------------------------------------
        # Step 8: Ranking
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        self._logger.info("[Step 8/8] Ranking candidates...")
        ranking_scorer = RankingScorer(self._config.scoring)
        ranked_df = ranking_scorer.rank(combined_df)
        self._logger.info(
            "[Step 8/8] Ranked %d candidates. Done in %.2fs",
            len(ranked_df), time.perf_counter() - t0,
        )

        # ------------------------------------------------------------------
        # Assemble result set
        # ------------------------------------------------------------------
        result_set = self._build_result_set(ranked_df)

        elapsed = time.perf_counter() - pipeline_start
        self._logger.info(
            "=== Pipeline complete: %d candidates in %.2fs ===",
            len(result_set), elapsed,
        )

        return result_set

    def _merge_wildtype_affinities(
        self, mutant_df: pd.DataFrame, wildtype_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge wildtype affinity values into the mutant prediction DataFrame.

        Aligns on (peptide, allele) — uses the mutant peptide sequence as
        the key from the candidate metadata and the allele column.

        Args:
            mutant_df: Mutant prediction DataFrame.
            wildtype_df: Wildtype prediction DataFrame (peptide = wt sequence).

        Returns:
            Mutant DataFrame with a ``wildtype_affinity`` column added.
        """
        if wildtype_df.empty or "wildtype_affinity" not in wildtype_df.columns:
            self._logger.warning(
                "Wildtype prediction DataFrame is empty or missing 'wildtype_affinity'"
            )
            mutant_df = mutant_df.copy()
            mutant_df["wildtype_affinity"] = float("nan")
            return mutant_df

        # The wildtype_df peptide column contains wildtype sequences;
        # we need to match them back to the mutant candidates by position.
        # For simplicity, merge on allele and positional index.
        wt_subset = wildtype_df[["peptide", "allele", "wildtype_affinity"]].rename(
            columns={"peptide": "wildtype_peptide_seq"}
        )

        # If mutant_df has a wildtype_peptide column from metadata, use it
        if "wildtype_sequence" in mutant_df.columns or "wildtype_peptide" in mutant_df.columns:
            wt_col = (
                "wildtype_peptide"
                if "wildtype_peptide" in mutant_df.columns
                else "wildtype_sequence"
            )
            merged = mutant_df.merge(
                wt_subset.rename(columns={"wildtype_peptide_seq": wt_col}),
                on=[wt_col, "allele"],
                how="left",
            )
        else:
            # Fallback: merge by position (assumes same ordering)
            mutant_df = mutant_df.copy()
            if len(wildtype_df) == len(mutant_df):
                mutant_df["wildtype_affinity"] = wildtype_df["wildtype_affinity"].values
            else:
                mutant_df["wildtype_affinity"] = float("nan")
            return mutant_df

        return merged

    def _attach_variant_metadata(
        self, df: pd.DataFrame, variants: list[SomaticVariant]
    ) -> pd.DataFrame:
        """Attach expression and VAF from the original variant list to the DataFrame.

        Matches on mutation_str (gene + protein_change) if that column exists,
        otherwise falls back to gene name matching.

        Args:
            df: Prediction DataFrame.
            variants: Original variant list.

        Returns:
            DataFrame with ``expression`` and ``vaf`` columns attached.
        """
        # Build a lookup: mutation_str -> (expression, vaf)
        variant_meta: dict[str, tuple[float, float]] = {}
        for v in variants:
            key = f"{v.gene}_{v.protein_change}"
            expression = v.expression if v.expression is not None else 0.0
            variant_meta[key] = (expression, v.vaf)

        df = df.copy()

        if "mutation_str" in df.columns:
            df["expression"] = df["mutation_str"].map(
                lambda m: variant_meta.get(m, (0.0, 0.0))[0]
            )
            df["vaf"] = df["mutation_str"].map(
                lambda m: variant_meta.get(m, (0.0, 0.0))[1]
            )
        else:
            # Fill with 0.0 if metadata is unavailable
            df["expression"] = 0.0
            df["vaf"] = 0.0
            self._logger.warning(
                "mutation_str column absent; expression and VAF set to 0.0"
            )

        return df

    def _build_result_set(self, ranked_df: pd.DataFrame) -> NeoantigenResultSet:
        """Convert the ranked DataFrame to a NeoantigenResultSet.

        Args:
            ranked_df: Fully scored and ranked prediction DataFrame.

        Returns:
            NeoantigenResultSet with one candidate per row.
        """
        candidates: list[NeoantigenCandidate] = []

        def _safe_float(val: object, default: float = 0.0) -> float:
            try:
                f = float(val)  # type: ignore[arg-type]
                return f if pd.notna(f) else default
            except (TypeError, ValueError):
                return default

        def _safe_str(val: object, default: str = "") -> str:
            return str(val) if pd.notna(val) else default

        for _, row in ranked_df.iterrows():
            gene = _safe_str(row.get("gene", ""))
            mutation = _safe_str(row.get("mutation_str", ""))

            # Determine best allele as the one with highest presentation score
            allele = _safe_str(row.get("allele", ""))

            candidate = NeoantigenCandidate(
                gene=gene,
                mutation=mutation,
                peptide=_safe_str(row.get("peptide", "")),
                wildtype_peptide=_safe_str(
                    row.get("wildtype_sequence", row.get("wildtype_peptide_seq", ""))
                ),
                best_allele=allele,
                presentation_score=_safe_float(row.get("mhcflurry_presentation_score")),
                binding_affinity_nm=_safe_float(row.get("mhcflurry_affinity")),
                wildtype_affinity_nm=_safe_float(row.get("wildtype_affinity")),
                processing_score=_safe_float(row.get("mhcflurry_processing_score")),
                agretopicity=_safe_float(row.get("agretopicity", 1.0)),
                expression=_safe_float(row.get("expression")),
                vaf=_safe_float(row.get("vaf")),
                composite_score=_safe_float(row.get("composite_score")),
                composite_rank=int(row.get("composite_rank", 0)),
            )
            candidates.append(candidate)

        return NeoantigenResultSet(candidates)
