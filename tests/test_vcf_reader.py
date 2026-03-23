"""Tests for VCF reading and somatic variant extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from neoantigen_pipeline.io.vcf_reader import (
    CONSEQUENCE_FRAMESHIFT,
    CONSEQUENCE_INFRAME_DEL,
    CONSEQUENCE_INFRAME_INS,
    CONSEQUENCE_MISSENSE,
    SomaticVariant,
    VCFReader,
    _parse_aa_sequence,
    _three_to_one,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_reader() -> VCFReader:
    """Return a VCFReader instance without opening a real file."""
    reader = VCFReader.__new__(VCFReader)
    reader._csq_fields = []
    return reader


def _make_variant(**kwargs) -> SomaticVariant:
    """Build a SomaticVariant with sensible defaults, overridden by kwargs."""
    defaults: dict = dict(
        chrom="7",
        pos=140453136,
        ref="A",
        alt="T",
        gene="BRAF",
        transcript_id="ENST00000288602",
        protein_change="p.Val600Glu",
        aa_ref="V",
        aa_alt="E",
        aa_pos=600,
        vaf=0.45,
        expression=12.3,
        consequence=CONSEQUENCE_MISSENSE,
        variant_type="missense",
    )
    defaults.update(kwargs)
    return SomaticVariant(**defaults)


# ── _three_to_one ─────────────────────────────────────────────────────────────


class TestThreeToOne:
    """Tests for the amino acid code converter."""

    def test_known_codes(self) -> None:
        assert _three_to_one("Val") == "V"
        assert _three_to_one("Glu") == "E"
        assert _three_to_one("Ala") == "A"

    def test_stop_codon(self) -> None:
        assert _three_to_one("Ter") == "*"

    def test_unknown_returns_original(self) -> None:
        assert _three_to_one("Xyz") == "Xyz"


# ── _parse_aa_sequence ────────────────────────────────────────────────────────


class TestParseAaSequence:
    """Tests for 3-letter sequence parser."""

    def test_single_code(self) -> None:
        assert _parse_aa_sequence("Gly") == "G"

    def test_two_codes(self) -> None:
        assert _parse_aa_sequence("GlyPro") == "GP"

    def test_three_codes(self) -> None:
        assert _parse_aa_sequence("GlyProVal") == "GPV"

    def test_stop_included(self) -> None:
        assert _parse_aa_sequence("ValTer") == "V*"


# ── SomaticVariant ────────────────────────────────────────────────────────────


class TestSomaticVariant:
    """Tests for the SomaticVariant dataclass."""

    def test_missense_creation(self) -> None:
        v = _make_variant()
        assert v.gene == "BRAF"
        assert v.aa_pos == 600
        assert v.vaf == pytest.approx(0.45)
        assert v.variant_type == "missense"
        assert v.is_frameshift is False
        assert v.downstream_sequence is None

    def test_expression_can_be_none(self) -> None:
        v = _make_variant(expression=None)
        assert v.expression is None

    def test_inframe_del_multi_char_aa_ref(self) -> None:
        v = _make_variant(
            protein_change="p.Val600_Lys601del",
            aa_ref="VK",
            aa_alt="",
            consequence=CONSEQUENCE_INFRAME_DEL,
            variant_type="inframe_deletion",
        )
        assert v.aa_ref == "VK"
        assert v.aa_alt == ""
        assert v.variant_type == "inframe_deletion"

    def test_frameshift_flags(self) -> None:
        v = _make_variant(
            protein_change="p.Val600Glyfs*12",
            aa_ref="V",
            aa_alt="G",
            consequence=CONSEQUENCE_FRAMESHIFT,
            variant_type="frameshift",
            is_frameshift=True,
            downstream_sequence="MGPQSTV",
        )
        assert v.is_frameshift is True
        assert v.downstream_sequence == "MGPQSTV"


# ── HGVSp parsing ─────────────────────────────────────────────────────────────


class TestHGVSpParsing:
    """Tests for VCFReader._parse_hgvsp covering all supported variant types."""

    def _reader(self) -> VCFReader:
        return _make_reader()

    # Missense
    def test_missense_standard(self) -> None:
        r = self._reader()._parse_hgvsp("p.Val600Glu")
        assert r is not None
        assert r.aa_ref == "V"
        assert r.aa_alt == "E"
        assert r.aa_pos == 600
        assert r.variant_type == "missense"
        assert r.is_frameshift is False

    def test_missense_with_transcript_prefix(self) -> None:
        r = self._reader()._parse_hgvsp("ENSP00000288602.7:p.Val600Glu")
        assert r is not None
        assert r.aa_ref == "V"
        assert r.aa_alt == "E"
        assert r.aa_pos == 600

    # Frameshift
    def test_frameshift_with_new_aa_and_stop(self) -> None:
        r = self._reader()._parse_hgvsp("p.Val600Glyfs*12")
        assert r is not None
        assert r.aa_ref == "V"
        assert r.aa_alt == "G"
        assert r.aa_pos == 600
        assert r.variant_type == "frameshift"
        assert r.is_frameshift is True

    def test_frameshift_without_new_aa(self) -> None:
        r = self._reader()._parse_hgvsp("p.Val600fs")
        assert r is not None
        assert r.aa_ref == "V"
        assert r.aa_alt == ""
        assert r.aa_pos == 600
        assert r.is_frameshift is True

    def test_frameshift_with_transcript_prefix(self) -> None:
        r = self._reader()._parse_hgvsp("ENSP00000:p.Glu118Valfs*3")
        assert r is not None
        assert r.aa_ref == "E"
        assert r.aa_alt == "V"
        assert r.aa_pos == 118
        assert r.is_frameshift is True

    def test_frameshift_unknown_stop_distance(self) -> None:
        r = self._reader()._parse_hgvsp("p.Ala23Serfs*?")
        assert r is not None
        assert r.aa_pos == 23
        assert r.is_frameshift is True

    # Inframe deletion – single residue
    def test_inframe_del_single(self) -> None:
        r = self._reader()._parse_hgvsp("p.Val600del")
        assert r is not None
        assert r.aa_ref == "V"
        assert r.aa_alt == ""
        assert r.aa_pos == 600
        assert r.variant_type == "inframe_deletion"
        assert r.is_frameshift is False

    def test_inframe_del_single_with_prefix(self) -> None:
        r = self._reader()._parse_hgvsp("ENSP00000:p.Lys12del")
        assert r is not None
        assert r.aa_ref == "K"
        assert r.aa_pos == 12

    # Inframe deletion – range
    def test_inframe_del_range(self) -> None:
        r = self._reader()._parse_hgvsp("p.Val600_Lys601del")
        assert r is not None
        assert r.aa_ref == "VK"
        assert r.aa_alt == ""
        assert r.aa_pos == 600
        assert r.variant_type == "inframe_deletion"

    def test_inframe_del_range_three_letter_codes(self) -> None:
        r = self._reader()._parse_hgvsp("p.Gly12_Ala14del")
        assert r is not None
        assert r.aa_ref == "GA"
        assert r.aa_pos == 12

    # Inframe insertion
    def test_inframe_ins_single_aa(self) -> None:
        r = self._reader()._parse_hgvsp("p.Ala600_Ala601insGly")
        assert r is not None
        assert r.aa_ref == ""
        assert r.aa_alt == "G"
        assert r.aa_pos == 600
        assert r.variant_type == "inframe_insertion"
        assert r.is_frameshift is False

    def test_inframe_ins_multiple_aas(self) -> None:
        r = self._reader()._parse_hgvsp("p.Ala600_Ala601insGlyPro")
        assert r is not None
        assert r.aa_alt == "GP"
        assert r.aa_pos == 600

    def test_inframe_ins_three_aas(self) -> None:
        r = self._reader()._parse_hgvsp("p.Lys5_Arg6insGlyProVal")
        assert r is not None
        assert r.aa_alt == "GPV"

    # Duplication (treated as inframe insertion)
    def test_dup_single_residue(self) -> None:
        r = self._reader()._parse_hgvsp("p.Val600dup")
        assert r is not None
        assert r.aa_ref == ""
        assert r.aa_alt == "V"
        assert r.aa_pos == 600
        assert r.variant_type == "inframe_insertion"

    # Unrecognised patterns
    def test_synonymous_notation_returns_none(self) -> None:
        # p.Val600= is synonymous (no change) — not a supported pattern
        assert self._reader()._parse_hgvsp("p.Val600=") is None

    def test_no_p_dot_returns_none(self) -> None:
        assert self._reader()._parse_hgvsp("c.1799T>A") is None

    def test_empty_returns_none(self) -> None:
        assert self._reader()._parse_hgvsp("") is None


# ── VCFReader internals ───────────────────────────────────────────────────────


class TestVCFReaderInternals:
    """Tests for VCFReader helper methods."""

    def test_parse_csq_entry(self) -> None:
        reader = _make_reader()
        reader._csq_fields = ["Allele", "Consequence", "SYMBOL", "Feature", "HGVSp"]
        entry = "T|missense_variant|BRAF|ENST00000288602|ENSP:p.Val600Glu"
        result = reader._parse_csq_entry(entry)
        assert result["SYMBOL"] == "BRAF"
        assert result["Consequence"] == "missense_variant"

    def test_field_index_found(self) -> None:
        reader = _make_reader()
        reader._csq_fields = ["Allele", "Consequence", "SYMBOL"]
        assert reader._field_index("SYMBOL") == 2

    def test_field_index_missing(self) -> None:
        reader = _make_reader()
        reader._csq_fields = ["Allele"]
        assert reader._field_index("MISSING") is None

    def test_match_consequence_simple(self) -> None:
        reader = _make_reader()
        result = reader._match_consequence("missense_variant", {CONSEQUENCE_MISSENSE})
        assert result == CONSEQUENCE_MISSENSE

    def test_match_consequence_compound(self) -> None:
        """VEP compound consequence (ampersand-separated) should match."""
        reader = _make_reader()
        result = reader._match_consequence(
            "missense_variant&splice_region_variant", {CONSEQUENCE_MISSENSE}
        )
        assert result == CONSEQUENCE_MISSENSE

    def test_match_consequence_no_match(self) -> None:
        reader = _make_reader()
        result = reader._match_consequence("synonymous_variant", {CONSEQUENCE_MISSENSE})
        assert result is None


# ── read_variants / read_missense_variants ────────────────────────────────────


def _mock_vcf_with_csq(csq_value: str, consequence_override: str | None = None):
    """Build a mock cyvcf2 VCF yielding one record with the given CSQ value."""
    mock_record = MagicMock()
    mock_record.CHROM = "7"
    mock_record.POS = 140453136
    mock_record.REF = "A"
    mock_record.ALT = ["T"]
    mock_record.INFO = {"CSQ": csq_value}
    mock_record.format.return_value = None

    mock_vcf = MagicMock()
    mock_vcf.__iter__ = MagicMock(return_value=iter([mock_record]))
    mock_vcf.header_iter.return_value = iter([])
    mock_vcf.close = MagicMock()
    return mock_vcf


_CSQ_FIELDS = [
    "Allele",
    "Consequence",
    "IMPACT",
    "SYMBOL",
    "Gene",
    "Feature_type",
    "Feature",
    "BIOTYPE",
    "EXON",
    "INTRON",
    "HGVSc",
    "HGVSp",
]


class TestReadVariants:
    """Integration-style tests for read_variants and read_missense_variants."""

    def _reader_with_fields(self) -> VCFReader:
        reader = VCFReader("fake.vcf")
        reader._csq_fields = _CSQ_FIELDS
        return reader

    def _csq(self, consequence: str, hgvsp: str) -> str:
        return (
            f"T|{consequence}|MODERATE|BRAF|ENSG00000157764|"
            f"Transcript|ENST00000288602|protein_coding|15/18||"
            f"c.1799T>A|{hgvsp}"
        )

    # Missense
    @patch("cyvcf2.VCF")
    def test_read_missense_variant(self, _mock_cls) -> None:
        reader = self._reader_with_fields()
        csq = self._csq("missense_variant", "ENSP:p.Val600Glu")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_missense_variants()
        assert len(variants) == 1
        v = variants[0]
        assert v.aa_ref == "V"
        assert v.aa_alt == "E"
        assert v.aa_pos == 600
        assert v.variant_type == "missense"
        assert v.is_frameshift is False

    @patch("cyvcf2.VCF")
    def test_non_missense_filtered_by_read_missense(self, _mock_cls) -> None:
        reader = self._reader_with_fields()
        csq = self._csq("synonymous_variant", "")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            assert reader.read_missense_variants() == []

    # Inframe deletion
    @patch("cyvcf2.VCF")
    def test_read_inframe_del_single(self, _mock_cls) -> None:
        reader = self._reader_with_fields()
        csq = self._csq("inframe_deletion", "ENSP:p.Val600del")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_variants({CONSEQUENCE_INFRAME_DEL})
        assert len(variants) == 1
        v = variants[0]
        assert v.aa_ref == "V"
        assert v.aa_alt == ""
        assert v.variant_type == "inframe_deletion"
        assert v.consequence == CONSEQUENCE_INFRAME_DEL

    @patch("cyvcf2.VCF")
    def test_read_inframe_del_range(self, _mock_cls) -> None:
        reader = self._reader_with_fields()
        csq = self._csq("inframe_deletion", "ENSP:p.Val600_Lys601del")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_variants({CONSEQUENCE_INFRAME_DEL})
        assert len(variants) == 1
        assert variants[0].aa_ref == "VK"

    # Inframe insertion
    @patch("cyvcf2.VCF")
    def test_read_inframe_ins(self, _mock_cls) -> None:
        reader = self._reader_with_fields()
        csq = self._csq("inframe_insertion", "ENSP:p.Ala600_Ala601insGlyPro")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_variants({CONSEQUENCE_INFRAME_INS})
        assert len(variants) == 1
        v = variants[0]
        assert v.aa_ref == ""
        assert v.aa_alt == "GP"
        assert v.variant_type == "inframe_insertion"

    # Frameshift
    @patch("cyvcf2.VCF")
    def test_read_frameshift(self, _mock_cls) -> None:
        reader = self._reader_with_fields()
        csq = self._csq("frameshift_variant", "ENSP:p.Val600Glyfs*12")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_variants({CONSEQUENCE_FRAMESHIFT})
        assert len(variants) == 1
        v = variants[0]
        assert v.aa_ref == "V"
        assert v.aa_alt == "G"
        assert v.aa_pos == 600
        assert v.variant_type == "frameshift"
        assert v.is_frameshift is True
        assert v.downstream_sequence is None  # not in CSQ fields here

    # Multi-type filter
    @patch("cyvcf2.VCF")
    def test_read_variants_multi_type_filter(self, _mock_cls) -> None:
        """read_variants with a multi-type filter only returns matching types."""
        reader = self._reader_with_fields()
        csq = self._csq("inframe_deletion", "ENSP:p.Ala12del")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            # Asking only for missense — should get nothing
            variants = reader.read_variants({CONSEQUENCE_MISSENSE})
        assert variants == []

    @patch("cyvcf2.VCF")
    def test_read_variants_default_includes_all_types(self, _mock_cls) -> None:
        """read_variants() with no filter returns any supported type."""
        reader = self._reader_with_fields()
        csq = self._csq("inframe_deletion", "ENSP:p.Ala12del")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_variants()
        assert len(variants) == 1
        assert variants[0].variant_type == "inframe_deletion"

    # Backward compatibility
    @patch("cyvcf2.VCF")
    def test_read_missense_variants_is_backward_compatible(self, _mock_cls) -> None:
        """read_missense_variants must still return SomaticVariant with variant_type."""
        reader = self._reader_with_fields()
        csq = self._csq("missense_variant", "ENSP:p.Arg175His")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_missense_variants()
        assert len(variants) == 1
        assert variants[0].variant_type == "missense"
        assert variants[0].consequence == CONSEQUENCE_MISSENSE

    # Regression: frameshift with immediate-stop HGVSp (p.Glu11Ter)
    @patch("cyvcf2.VCF")
    def test_frameshift_with_immediate_stop_hgvsp(self, _mock_cls) -> None:
        """A frameshift_variant consequence must yield variant_type='frameshift'
        even when its HGVSp (e.g. p.Glu11Ter) looks structurally like a missense.

        Regression for the bug where _parse_hgvsp classified p.Glu11Ter as
        'missense' because 'Ter' matched the missense regex, and that heuristic
        was incorrectly used instead of the VEP Consequence field.
        """
        reader = self._reader_with_fields()
        # p.Glu11Ter: frameshift that hits a stop codon at position 11.
        # The HGVSp has no 'fs' marker, so _parse_hgvsp's heuristic returns
        # variant_type='missense' — but the Consequence field says otherwise.
        csq = self._csq("frameshift_variant", "ENSP:p.Glu11Ter")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_variants({CONSEQUENCE_FRAMESHIFT})
        assert len(variants) == 1
        v = variants[0]
        assert v.variant_type == "frameshift", (
            f"Expected 'frameshift', got '{v.variant_type}'. "
            "variant_type must be derived from VEP Consequence, not HGVSp pattern."
        )
        assert v.is_frameshift is True
        assert v.consequence == CONSEQUENCE_FRAMESHIFT
        # Coordinates are still correctly extracted from HGVSp
        assert v.aa_ref == "E"
        assert v.aa_alt == "*"
        assert v.aa_pos == 11

    @patch("cyvcf2.VCF")
    def test_frameshift_excluded_from_read_missense_variants(self, _mock_cls) -> None:
        """A frameshift_variant consequence must not appear in read_missense_variants(),
        even when its HGVSp could match the missense pattern."""
        reader = self._reader_with_fields()
        csq = self._csq("frameshift_variant", "ENSP:p.Glu11Ter")
        mock_vcf = _mock_vcf_with_csq(csq)
        with patch.object(reader, "_open_vcf", return_value=mock_vcf):
            variants = reader.read_missense_variants()
        assert variants == [], (
            "frameshift_variant must never appear in read_missense_variants() output."
        )
