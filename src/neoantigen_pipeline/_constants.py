"""Shared constants for the neoantigen prediction pipeline.

Centralises all values that would otherwise be duplicated across modules:
amino-acid alphabets, default parameter values, and DataFrame column names.
Import from here rather than re-defining locally.
"""

from __future__ import annotations

# ── Amino acid alphabet ──────────────────────────────────────────────────────

#: The canonical 20 standard amino acids (single-letter codes).
STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# ── Pipeline defaults ─────────────────────────────────────────────────────────

#: Default k-mer lengths for both peptide tiling and MHC-I prediction.
DEFAULT_PEPTIDE_LENGTHS: tuple[int, ...] = (8, 9, 10, 11)

#: Default length of N- and C-terminal flanking sequences for processing models.
DEFAULT_FLANK_LENGTH: int = 10

#: Default self-similarity threshold above which a peptide is considered self-like.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.8

# ── DataFrame column names ────────────────────────────────────────────────────
# Single source of truth for all column names used across pipeline DataFrames.

# Core peptide / candidate identity
COL_PEPTIDE: str = "peptide"
COL_MUT_PEPTIDE: str = "mut_peptide"
COL_BEST_ALLELE: str = "best_allele"
COL_PEPTIDE_NUM: str = "peptide_num"
COL_SAMPLE_NAME: str = "sample_name"
COL_N_FLANK: str = "n_flank"
COL_C_FLANK: str = "c_flank"

# Candidate metadata from variant annotation
COL_MUTATION_STR: str = "mutation_str"
COL_GENE: str = "gene"
COL_TRANSCRIPT_ID: str = "transcript_id"
COL_AA_POS: str = "aa_pos"

# MHCflurry raw output columns (before standardisation)
_RAW_AFFINITY: str = "affinity"
_RAW_AFFINITY_PERCENTILE: str = "affinity_percentile"
_RAW_PROCESSING_SCORE: str = "processing_score"
_RAW_PRESENTATION_SCORE: str = "presentation_score"

# Standardised MHCflurry score columns
COL_MHCFLURRY_AFFINITY: str = "mhcflurry_affinity"
COL_AFFINITY_PERCENTILE: str = "mhcflurry_affinity_percentile"
COL_PROCESSING_SCORE: str = "mhcflurry_processing_score"
COL_PRESENTATION_SCORE: str = "mhcflurry_presentation_score"
COL_PRESENTATION_PERCENTILE: str = "presentation_percentile"

#: Map from raw MHCflurry column names to standardised pipeline names.
MHCFLURRY_RENAME_MAP: dict[str, str] = {
    _RAW_AFFINITY: COL_MHCFLURRY_AFFINITY,
    _RAW_AFFINITY_PERCENTILE: COL_AFFINITY_PERCENTILE,
    _RAW_PROCESSING_SCORE: COL_PROCESSING_SCORE,
    _RAW_PRESENTATION_SCORE: COL_PRESENTATION_SCORE,
}

# Wildtype and derived scoring columns
COL_WILDTYPE_AFFINITY: str = "wildtype_affinity"
COL_AGRETOPICITY: str = "agretopicity"

# Variant-level annotation columns
COL_EXPRESSION: str = "expression"
COL_VAF: str = "vaf"

# Composite ranking columns
COL_COMPOSITE_SCORE: str = "composite_score"
COL_COMPOSITE_RANK: str = "composite_rank"
