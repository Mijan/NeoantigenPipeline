"""Peptide tiling from mutant protein sequences.

Generates all k-mer windows that span a point mutation, along with their
corresponding wildtype equivalents and flanking sequences for MHC processing
prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from neoantigen_pipeline.exceptions import ProteomeError

if TYPE_CHECKING:
    from neoantigen_pipeline.config import PeptideGenerationConfig
    from neoantigen_pipeline.io.proteome import ProteomeDB
    from neoantigen_pipeline.io.vcf_reader import SomaticVariant


@dataclass
class PeptideCandidate:
    """A single tiled peptide candidate derived from a missense mutation.

    This dataclass uses __slots__ for memory efficiency since many thousands
    of candidates may be generated per run.

    Attributes:
        peptide_sequence: Mutant k-mer amino acid sequence.
        wildtype_sequence: Corresponding wildtype k-mer at the same position.
        gene: HGNC gene symbol.
        mutation_str: Human-readable mutation label (e.g. "BRAF_p.Val600Glu").
        transcript_id: Ensembl transcript identifier.
        aa_pos: 1-based position of the mutated residue in the full protein.
        position_in_peptide: 0-based index of the mutation within the k-mer.
        n_flank: N-terminal flanking sequence (length = config.n_flank_length).
        c_flank: C-terminal flanking sequence (length = config.c_flank_length).
        peptide_length: Length of the k-mer (k).
    """

    __slots__ = (
        "peptide_sequence",
        "wildtype_sequence",
        "gene",
        "mutation_str",
        "transcript_id",
        "aa_pos",
        "position_in_peptide",
        "n_flank",
        "c_flank",
        "peptide_length",
    )

    peptide_sequence: str
    wildtype_sequence: str
    gene: str
    mutation_str: str
    transcript_id: str
    aa_pos: int
    position_in_peptide: int
    n_flank: str
    c_flank: str
    peptide_length: int


class PeptideGenerator:
    """Generates mutant and wildtype k-mer peptide candidates from somatic variants.

    For each variant, the generator:
    1. Retrieves the wildtype protein sequence from the reference proteome.
    2. Applies the amino acid substitution to produce the mutant sequence.
    3. Tiles all windows of each configured length that contain the mutation.
    4. Extracts N- and C-terminal flanking sequences for processing prediction.

    Args:
        config: Peptide generation configuration.
        proteome_db: Reference proteome database for wildtype sequence lookup.
    """

    def __init__(
        self,
        config: PeptideGenerationConfig,
        proteome_db: ProteomeDB,
    ) -> None:
        self._config = config
        self._proteome_db = proteome_db
        self._logger = logging.getLogger(type(self).__qualname__)

    def generate(self, variant: SomaticVariant) -> list[PeptideCandidate]:
        """Generate all peptide candidates for a single somatic variant.

        Retrieves the wildtype protein, applies the substitution, and tiles
        k-mer windows of each configured length across the mutation site.

        Args:
            variant: A missense somatic variant with aa_pos, aa_ref, aa_alt,
                transcript_id, gene, and protein_change attributes.

        Returns:
            List of PeptideCandidate objects. Empty if the transcript is not
            found in the proteome or if the mutation coordinates are out of range.

        Raises:
            ProteomeError: If a proteome lookup failure prevents generation.
        """
        wt_protein = self._proteome_db.get_sequence(variant.transcript_id)

        if wt_protein is None:
            self._logger.warning(
                "Transcript '%s' not found in proteome; skipping %s",
                variant.transcript_id, variant.protein_change,
            )
            return []

        # Validate amino acid position (1-based → 0-based index)
        mut_idx = variant.aa_pos - 1
        if mut_idx < 0 or mut_idx >= len(wt_protein):
            self._logger.warning(
                "Mutation position %d out of range for transcript '%s' (length %d)",
                variant.aa_pos, variant.transcript_id, len(wt_protein),
            )
            return []

        # Verify reference amino acid matches the proteome
        actual_ref = wt_protein[mut_idx]
        if actual_ref != variant.aa_ref:
            self._logger.warning(
                "Reference mismatch at %s pos %d: expected '%s', found '%s' in proteome",
                variant.transcript_id, variant.aa_pos, variant.aa_ref, actual_ref,
            )
            # Continue anyway — annotation may differ from proteome version

        # Build mutant protein
        mut_protein = wt_protein[:mut_idx] + variant.aa_alt + wt_protein[mut_idx + 1:]

        mutation_str = f"{variant.gene}_{variant.protein_change}"
        candidates: list[PeptideCandidate] = []

        for k in self._config.peptide_lengths:
            candidates.extend(
                self._tile_windows(
                    wt_protein=wt_protein,
                    mut_protein=mut_protein,
                    mut_idx=mut_idx,
                    k=k,
                    gene=variant.gene,
                    mutation_str=mutation_str,
                    transcript_id=variant.transcript_id,
                    aa_pos=variant.aa_pos,
                )
            )

        return candidates

    def _tile_windows(
        self,
        wt_protein: str,
        mut_protein: str,
        mut_idx: int,
        k: int,
        gene: str,
        mutation_str: str,
        transcript_id: str,
        aa_pos: int,
    ) -> list[PeptideCandidate]:
        """Generate all k-mer windows spanning the mutation position.

        A window of length k spans the mutation when the mutation index falls
        within [window_start, window_start + k - 1].

        Args:
            wt_protein: Full wildtype protein sequence.
            mut_protein: Full mutant protein sequence.
            mut_idx: 0-based index of the mutated residue.
            k: Peptide length.
            gene: Gene symbol.
            mutation_str: Mutation label string.
            transcript_id: Transcript identifier.
            aa_pos: 1-based amino acid position.

        Returns:
            List of PeptideCandidate instances for this k and mutation.
        """
        protein_len = len(wt_protein)
        if k > protein_len:
            return []

        # Window start range: mut_idx must be in [start, start+k-1]
        # => start in [mut_idx - k + 1, mut_idx]
        first_start = max(0, mut_idx - k + 1)
        last_start = min(protein_len - k, mut_idx)

        candidates: list[PeptideCandidate] = []

        for start in range(first_start, last_start + 1):
            end = start + k
            mut_pep = mut_protein[start:end]
            wt_pep = wt_protein[start:end]

            # Skip if mutant and wildtype are identical (shouldn't happen, but defensive)
            if mut_pep == wt_pep:
                continue

            # Extract flanking sequences (clamped to protein boundaries)
            n_flank = self._extract_n_flank(wt_protein, start)
            c_flank = self._extract_c_flank(wt_protein, end, protein_len)

            position_in_peptide = mut_idx - start

            candidates.append(
                PeptideCandidate(
                    peptide_sequence=mut_pep,
                    wildtype_sequence=wt_pep,
                    gene=gene,
                    mutation_str=mutation_str,
                    transcript_id=transcript_id,
                    aa_pos=aa_pos,
                    position_in_peptide=position_in_peptide,
                    n_flank=n_flank,
                    c_flank=c_flank,
                    peptide_length=k,
                )
            )

        return candidates

    def _extract_n_flank(self, protein: str, pep_start: int) -> str:
        """Extract the N-terminal flanking sequence before the peptide window.

        Args:
            protein: Full protein sequence.
            pep_start: 0-based start index of the peptide window.

        Returns:
            Up to n_flank_length amino acids immediately before pep_start.
        """
        n_len = self._config.n_flank_length
        flank_start = max(0, pep_start - n_len)
        return protein[flank_start:pep_start]

    def _extract_c_flank(self, protein: str, pep_end: int, protein_len: int) -> str:
        """Extract the C-terminal flanking sequence after the peptide window.

        Args:
            protein: Full protein sequence.
            pep_end: 0-based exclusive end index of the peptide window.
            protein_len: Total protein length.

        Returns:
            Up to c_flank_length amino acids immediately after pep_end.
        """
        c_len = self._config.c_flank_length
        flank_end = min(protein_len, pep_end + c_len)
        return protein[pep_end:flank_end]
