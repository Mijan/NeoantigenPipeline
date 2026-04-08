"""Tests for peptide generation from somatic variants."""

from __future__ import annotations

from unittest.mock import MagicMock

from neoantigen_pipeline.candidates.peptide_generator import (
    PeptideCandidate,
    PeptideGenerator,
)
from neoantigen_pipeline.config import PeptideGenerationConfig
from neoantigen_pipeline.io.vcf_reader import SomaticVariant


def _make_variant(
    aa_ref: str = "V",
    aa_alt: str = "E",
    aa_pos: int = 600,
    gene: str = "BRAF",
    transcript_id: str = "ENST00000288602",
    variant_type: str = "missense",
    consequence: str = "missense_variant",
) -> SomaticVariant:
    return SomaticVariant(
        chrom="7",
        pos=140453136,
        ref="A",
        alt="T",
        gene=gene,
        transcript_id=transcript_id,
        variant_type=variant_type,
        protein_change=f"p.{aa_ref}{aa_pos}{aa_alt}",
        aa_ref=aa_ref,
        aa_alt=aa_alt,
        aa_pos=aa_pos,
        vaf=0.45,
        expression=10.0,
        consequence=consequence,
    )


def _make_proteome(sequence: str, transcript_id: str = "ENST00000288602") -> MagicMock:
    db = MagicMock()
    db.get_sequence.return_value = sequence
    db._sequences = {transcript_id: sequence}
    return db


class TestPeptideCandidate:
    """Tests for the PeptideCandidate dataclass."""

    def test_slots_defined(self):
        assert hasattr(PeptideCandidate, "__slots__")

    def test_creation(self):
        candidate = PeptideCandidate(
            peptide_sequence="FLEELIQEF",
            wildtype_sequence="FLVVLIQEF",
            gene="BRAF",
            mutation_str="BRAF_p.Val600Glu",
            transcript_id="ENST00000288602",
            aa_pos=600,
            position_in_peptide=2,
            n_flank="QSSSAPGN",
            c_flank="PDFIQQ",
            peptide_length=9,
        )
        assert candidate.peptide_length == 9
        assert candidate.position_in_peptide == 2


class TestPeptideGenerator:
    """Tests for PeptideGenerator tiling logic."""

    def _make_generator(self, lengths=(9,), n_flank=5, c_flank=5) -> PeptideGenerator:
        config = PeptideGenerationConfig(
            peptide_lengths=lengths,
            n_flank_length=n_flank,
            c_flank_length=c_flank,
        )
        proteome = _make_proteome("A" * 620, "ENST00000288602")
        return PeptideGenerator(config, proteome)

    def test_missing_transcript_returns_empty(self):
        config = PeptideGenerationConfig(peptide_lengths=(9,))
        proteome = MagicMock()
        proteome.get_sequence.return_value = None
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant()
        result = gen.generate(variant)
        assert result == []

    def test_out_of_range_position_returns_empty(self):
        config = PeptideGenerationConfig(peptide_lengths=(9,))
        protein = "ACDEFGHIK"  # length 9
        proteome = _make_proteome(protein)
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_pos=100)  # beyond protein length
        result = gen.generate(variant)
        assert result == []

    def test_generates_all_windows_containing_mutation(self):
        # 30aa protein, mutation at position 15 (1-based)
        protein = "AAAAAAAAAAAAAAACAAAAAAAAAAAAAA"
        assert len(protein) == 30
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="C", aa_alt="G", aa_pos=16)
        candidates = gen.generate(variant)

        # All windows of length 9 that overlap position 15 (0-based index 15)
        # start positions: max(0, 15-8)=7 to min(21, 15)=15 => 9 windows
        assert len(candidates) == 9

    def test_peptide_contains_mutant_aa(self):
        protein = "AAAAAAAAAAAAAAACAAAAAAAAAAAAAA"
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="C", aa_alt="G", aa_pos=16)
        candidates = gen.generate(variant)
        for c in candidates:
            assert "G" in c.peptide_sequence, (
                f"Mutant aa 'G' not in {c.peptide_sequence}"
            )

    def test_wildtype_peptide_contains_ref_aa(self):
        protein = "AAAAAAAAAAAAAAACAAAAAAAAAAAAAA"
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="C", aa_alt="G", aa_pos=16)
        candidates = gen.generate(variant)
        for c in candidates:
            assert "C" in c.wildtype_sequence

    def test_mutation_at_protein_start(self):
        """Mutation at position 1 should produce windows from the start."""
        protein = "MADTEVSGNL"  # 10aa
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(5,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="M", aa_alt="V", aa_pos=1)
        candidates = gen.generate(variant)
        # Only 1 window of length 5 can start at position 0 (mutation at idx 0)
        assert len(candidates) >= 1
        assert candidates[0].peptide_sequence[0] == "V"

    def test_mutation_at_protein_end(self):
        """Mutation at the last position should produce windows ending at the end."""
        protein = "MADTEVSGNL"  # length 10
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(5,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="L", aa_alt="P", aa_pos=10)
        candidates = gen.generate(variant)
        assert len(candidates) >= 1
        for c in candidates:
            assert c.peptide_sequence[-1] == "P" or "P" in c.peptide_sequence

    def test_n_flank_extraction(self):
        protein = "AAAAAAAAAA" + "C" + "AAAAAAAAAA"  # 21aa, C at position 11
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=4, c_flank_length=4
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="C", aa_alt="G", aa_pos=11)
        candidates = gen.generate(variant)
        for c in candidates:
            assert len(c.n_flank) <= 4
            assert len(c.c_flank) <= 4

    def test_multiple_peptide_lengths(self):
        protein = "AAAAAAAAAAAAAAACAAAAAAAAAAAAAA"
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(8, 9, 10), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(aa_ref="C", aa_alt="G", aa_pos=16)
        candidates = gen.generate(variant)
        lengths = {c.peptide_length for c in candidates}
        assert lengths == {8, 9, 10}


class TestInframeIndels:
    """Tests for inframe insertion and deletion handling."""

    def _make_generator(self, lengths=(9,)) -> PeptideGenerator:
        config = PeptideGenerationConfig(
            peptide_lengths=lengths, n_flank_length=0, c_flank_length=0
        )
        # 30aa protein: positions 1-30
        protein = "MADTEFGHIKLMNPQRSTVWYACDEFGHI"  # 29aa
        protein = protein + "K"  # 30aa
        proteome = _make_proteome(protein)
        return PeptideGenerator(config, proteome), protein

    def test_single_residue_inframe_deletion(self):
        """Single-residue deletion: mutant is 1aa shorter than wildtype."""
        # 20aa protein with V at position 10
        protein = "ACDEFGHIKL" + "V" + "MNPQRSTVWY"
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(
            aa_ref="V",
            aa_alt="",
            aa_pos=11,
            variant_type="inframe_deletion",
            consequence="inframe_deletion",
        )
        candidates = gen.generate(variant)
        assert len(candidates) > 0
        # Mutant peptides should be length 9; wildtype peptides also 9
        for c in candidates:
            assert len(c.peptide_sequence) == 9
            assert len(c.wildtype_sequence) == 9
        # Mutant and wildtype sequences should differ (deletion shifts sequence)
        peptide_seqs = {c.peptide_sequence for c in candidates}
        wt_seqs = {c.wildtype_sequence for c in candidates}
        assert peptide_seqs != wt_seqs

    def test_multi_residue_inframe_deletion(self):
        """Two-residue deletion: mutant is 2aa shorter; reference validation passes."""
        # 25aa protein with VK at positions 10-11
        protein = "ACDEFGHIKL" + "VK" + "MNPQRSTVWYACDE"
        assert len(protein) == 26
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        # "VK" is at 1-based positions 11-12 (0-based indices 10-11)
        variant = _make_variant(
            aa_ref="VK",
            aa_alt="",
            aa_pos=11,
            variant_type="inframe_deletion",
            consequence="inframe_deletion",
        )
        candidates = gen.generate(variant)
        assert len(candidates) > 0
        # Mutant protein is 2aa shorter: check no short peptides were produced
        for c in candidates:
            assert len(c.peptide_sequence) == 9
            assert len(c.wildtype_sequence) == 9
        # Verify mutant sequences differ from wildtype
        assert any(c.peptide_sequence != c.wildtype_sequence for c in candidates)

    def test_inframe_insertion(self):
        """Insertion of two residues: mutant is longer than wildtype."""
        # 20aa protein; insertion of GP after position 10
        protein = "ACDEFGHIKL" + "MNPQRSTVWY"
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(
            peptide_lengths=(9,), n_flank_length=0, c_flank_length=0
        )
        gen = PeptideGenerator(config, proteome)
        # Insertion of "GP" between position 10 and 11 (aa_pos=10)
        variant = _make_variant(
            aa_ref="",
            aa_alt="GP",
            aa_pos=10,
            variant_type="inframe_insertion",
            consequence="inframe_insertion",
        )
        candidates = gen.generate(variant)
        assert len(candidates) > 0
        # Mutant protein is 2aa longer; 9-mer windows should still work
        for c in candidates:
            assert len(c.peptide_sequence) == 9
            assert len(c.wildtype_sequence) == 9
        # At least one mutant window should contain "G" or "P" (inserted residues)
        assert any(
            "G" in c.peptide_sequence or "P" in c.peptide_sequence for c in candidates
        )

    def test_deletion_reference_validation_passes(self):
        """No spurious reference-mismatch warning for correctly parsed multi-char aa_ref."""
        import logging

        # "VK" is at 1-based positions 11-12 (0-based indices 10-11)
        protein = "ACDEFGHIKL" + "VK" + "MNPQRSTVWYACDE"
        proteome = _make_proteome(protein)
        config = PeptideGenerationConfig(peptide_lengths=(9,))
        gen = PeptideGenerator(config, proteome)
        variant = _make_variant(
            aa_ref="VK",
            aa_alt="",
            aa_pos=11,
            variant_type="inframe_deletion",
            consequence="inframe_deletion",
        )

        # Capture warning records from the generator's logger
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture()
        logger = logging.getLogger("PeptideGenerator")
        logger.addHandler(handler)
        try:
            gen.generate(variant)
        finally:
            logger.removeHandler(handler)

        mismatch = [r for r in records if "Reference mismatch" in r.getMessage()]
        assert not mismatch, (
            f"Unexpected reference-mismatch warning: {mismatch[0].getMessage()}"
        )
