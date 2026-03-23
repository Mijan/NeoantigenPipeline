"""Typed result dataclasses for MHC binding predictions.

Replaces untyped DataFrames at all public prediction API boundaries.
All classes use ``__slots__`` and ``frozen=True`` for memory efficiency
and immutability — predictions must not be mutated after creation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BindingPrediction:
    """Minimal MHC binding prediction from the base ``predict()`` interface.

    Produced when only peptide sequences and alleles are available (no
    candidate metadata). Subclasses add further fields.

    Attributes:
        peptide: Amino acid sequence of the query peptide.
        best_allele: HLA allele with the highest presentation score.
        affinity_nm: Predicted IC50 binding affinity in nM (lower = tighter).
        affinity_percentile: Affinity percentile rank (lower = stronger binder).
        processing_score: Antigen processing score from the predictor.
        presentation_score: Combined MHC binding + processing score.
        presentation_percentile: Presentation percentile rank.
    """

    peptide: str
    best_allele: str
    affinity_nm: float
    affinity_percentile: float
    processing_score: float
    presentation_score: float
    presentation_percentile: float


@dataclass(frozen=True, slots=True)
class MHCIPredictionResult:
    """Full MHC-I prediction result with candidate metadata.

    Produced by ``MHCIPredictor.predict_with_processing()``. One instance
    per input ``PeptideCandidate`` (best allele selected by the predictor).

    Attributes:
        peptide: Mutant amino acid sequence.
        best_allele: HLA allele with the highest presentation score.
        affinity_nm: Predicted IC50 binding affinity in nM.
        affinity_percentile: Affinity percentile rank.
        processing_score: Antigen processing score.
        presentation_score: Combined MHC binding + processing score.
        presentation_percentile: Presentation percentile rank.
        mutation_str: Human-readable mutation label (e.g. "BRAF_p.Val600Glu").
        transcript_id: Ensembl transcript identifier.
        gene: HGNC gene symbol.
        aa_pos: 1-based position of the mutated residue in the full protein.
        n_flank: N-terminal flanking sequence used for processing prediction.
        c_flank: C-terminal flanking sequence used for processing prediction.
    """

    peptide: str
    best_allele: str
    affinity_nm: float
    affinity_percentile: float
    processing_score: float
    presentation_score: float
    presentation_percentile: float
    mutation_str: str
    transcript_id: str
    gene: str
    aa_pos: int
    n_flank: str
    c_flank: str


@dataclass(frozen=True, slots=True)
class WildtypePredictionResult:
    """Wildtype binding prediction paired by index with ``MHCIPredictionResult``.

    Produced by ``MHCIPredictor.predict_wildtype()``. The result at list
    index *i* always corresponds to the ``MHCIPredictionResult`` at index *i*
    — both derive from the same ``PeptideCandidate``.

    Attributes:
        wt_peptide: Wildtype amino acid sequence.
        mut_peptide: Mutant sequence of the paired candidate (for verification).
        wildtype_affinity_nm: Predicted IC50 binding affinity of the wildtype
            peptide in nM. Used to compute agretopicity.
    """

    wt_peptide: str
    mut_peptide: str
    wildtype_affinity_nm: float


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """Intermediate candidate with all scores computed, before composite ranking.

    Assembled by the pipeline from a ``MHCIPredictionResult``,
    a ``WildtypePredictionResult``, and variant-level annotation.
    Passed to ``RankingScorer`` to produce the final ``NeoantigenCandidate``
    objects with composite scores and ranks.

    Attributes:
        peptide: Mutant peptide amino acid sequence.
        wildtype_peptide: Corresponding wildtype peptide sequence.
        gene: HGNC gene symbol.
        mutation_str: Human-readable mutation label.
        best_allele: HLA allele with the highest presentation score.
        presentation_score: MHCflurry combined presentation score (0–1).
        binding_affinity_nm: Mutant IC50 in nM.
        wildtype_affinity_nm: Wildtype IC50 in nM.
        processing_score: Antigen processing score.
        agretopicity: Ratio of wildtype to mutant affinity (>1 = mutant binds stronger).
        expression: Gene expression level in TPM/FPKM.
        vaf: Variant allele fraction (0–1).
    """

    peptide: str
    wildtype_peptide: str
    gene: str
    mutation_str: str
    best_allele: str
    presentation_score: float
    binding_affinity_nm: float
    wildtype_affinity_nm: float
    processing_score: float
    agretopicity: float
    expression: float
    vaf: float
