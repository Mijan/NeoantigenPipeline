"""VCF reading and somatic variant extraction.

Parses VEP-annotated VCF files using cyvcf2 and extracts SomaticVariant
records for missense SNVs, inframe indels, and frameshift mutations.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cyvcf2

from neoantigen_pipeline.exceptions import VCFParsingError

if TYPE_CHECKING:
    pass

# ── Consequence constants ────────────────────────────────────────────────────

CONSEQUENCE_MISSENSE: str = "missense_variant"
CONSEQUENCE_INFRAME_INS: str = "inframe_insertion"
CONSEQUENCE_INFRAME_DEL: str = "inframe_deletion"
CONSEQUENCE_FRAMESHIFT: str = "frameshift_variant"

SUPPORTED_CONSEQUENCES: frozenset[str] = frozenset({
    CONSEQUENCE_MISSENSE,
    CONSEQUENCE_INFRAME_INS,
    CONSEQUENCE_INFRAME_DEL,
    CONSEQUENCE_FRAMESHIFT,
})

_CONSEQUENCE_TO_VTYPE: dict[str, str] = {
    CONSEQUENCE_MISSENSE: "missense",
    CONSEQUENCE_INFRAME_INS: "inframe_insertion",
    CONSEQUENCE_INFRAME_DEL: "inframe_deletion",
    CONSEQUENCE_FRAMESHIFT: "frameshift",
}

# ── CSQ field defaults ───────────────────────────────────────────────────────

_DEFAULT_CSQ_FIELDS: list[str] = [
    "Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type",
    "Feature", "BIOTYPE", "EXON", "INTRON", "HGVSc", "HGVSp",
    "cDNA_position", "CDS_position", "Protein_position", "Amino_acids",
    "Codons", "Existing_variation", "DISTANCE", "STRAND", "FLAGS",
    "SYMBOL_SOURCE", "HGNC_ID",
]

# ── Amino acid lookup ────────────────────────────────────────────────────────

_AA_3TO1: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U", "Pyl": "O",
}

# ── Compiled HGVSp regexes ───────────────────────────────────────────────────

_RE_MISSENSE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
_RE_FRAMESHIFT = re.compile(
    r"^p\.([A-Z][a-z]{2})(\d+)(?:([A-Z][a-z]{2}))?fs(?:\*(\d+|\?))?$"
)
_RE_DEL_SINGLE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)del$")
_RE_DEL_RANGE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)_([A-Z][a-z]{2})(\d+)del$")
_RE_DUP_SINGLE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)dup$")
_RE_INS = re.compile(
    r"^p\.([A-Z][a-z]{2})(\d+)_([A-Z][a-z]{2})(\d+)ins([A-Za-z]+)$"
)


def _three_to_one(three: str) -> str:
    """Convert a 3-letter amino acid code to single-letter.

    Args:
        three: Three-letter code, e.g. "Val".

    Returns:
        Single-letter code, or the original string if not found.
    """
    return _AA_3TO1.get(three, three)


def _parse_aa_sequence(seq: str) -> str:
    """Parse a run of 3-letter amino acid codes to single-letter codes.

    Args:
        seq: Concatenated 3-letter codes, e.g. "GlyPro".

    Returns:
        Single-letter string, e.g. "GP".
    """
    return "".join(_three_to_one(seq[i: i + 3]) for i in range(0, len(seq) - 2, 3))


# ── Internal HGVSp result ────────────────────────────────────────────────────

@dataclass
class _ParsedHGVSp:
    """Internal parsed representation of an HGVSp notation string."""

    aa_ref: str
    aa_alt: str
    aa_pos: int
    variant_type: str
    is_frameshift: bool = False


# ── Public data class ────────────────────────────────────────────────────────

@dataclass
class SomaticVariant:
    """A somatic variant extracted from a VEP-annotated VCF record.

    Supports missense SNVs, inframe indels, and frameshifts.

    Attributes:
        chrom: Chromosome name.
        pos: 1-based genomic position.
        ref: Reference allele.
        alt: Alternate allele.
        gene: HGNC gene symbol.
        transcript_id: Ensembl transcript identifier (e.g. ENST00000...).
        protein_change: HGVS protein change notation (e.g. "p.Val600Glu").
        aa_ref: Reference amino acid(s) in single-letter code.
            Multi-character for range deletions (e.g. "VK" for Val600_Lys601del).
            Empty string for pure insertions.
        aa_alt: Alternate amino acid(s) in single-letter code.
            Multi-character for insertions. Empty string for pure deletions.
        aa_pos: 1-based position of the change start in the protein.
        vaf: Variant allele fraction (0.0–1.0).
        expression: Gene expression in TPM/FPKM, or None if not annotated.
        consequence: VEP consequence term (e.g. "missense_variant").
        variant_type: Normalised type label: one of "missense",
            "inframe_insertion", "inframe_deletion", "frameshift".
        is_frameshift: True when this is a frameshift_variant.
        downstream_sequence: Novel frameshifted protein sequence from the
            VEP FrameshiftSequence plugin field, or None if unavailable.
        wildtype_protein_sequence: Full wildtype protein sequence from the
            VEP WildtypeProtein plugin field, or None if unavailable.
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    transcript_id: str
    protein_change: str
    aa_ref: str
    aa_alt: str
    aa_pos: int
    vaf: float
    expression: float | None
    consequence: str
    variant_type: str
    is_frameshift: bool = False
    downstream_sequence: str | None = None
    wildtype_protein_sequence: str | None = None


# ── Reader ───────────────────────────────────────────────────────────────────

class VCFReader:
    """Reads VEP-annotated VCF files and extracts somatic variants.

    Supports missense SNVs, inframe insertions/deletions, and frameshifts.
    Uses cyvcf2 for fast, streaming VCF access and parses the VEP CSQ INFO
    field to extract gene, transcript, and protein-level annotations.

    Expression values are read from FORMAT fields GX (gene-level) and TX
    (transcript-level), which are populated by pVACtools-style annotation
    pipelines. VAF is extracted from the tumour sample FORMAT.AF field.

    Args:
        vcf_path: Path to the (optionally gzipped, tabix-indexed) VCF file.
        tumor_sample_index: 0-based index of the tumour sample in the VCF.
            For paired tumour-normal VCFs (e.g. HCC1395 with samples
            [NORMAL, TUMOR]) this is 1. For single-sample VCFs use 0.

    Raises:
        VCFParsingError: If the file cannot be opened or lacks CSQ annotations.
    """

    def __init__(self, vcf_path: str, tumor_sample_index: int = 1) -> None:
        self._vcf_path = vcf_path
        self._tumor_sample_index = tumor_sample_index
        self._logger = logging.getLogger(type(self).__qualname__)
        self._csq_fields: list[str] = []

    # ── VCF access ──────────────────────────────────────────────────────────

    def _open_vcf(self) -> cyvcf2.VCF:
        """Open the VCF and parse the CSQ header.

        Returns:
            An open cyvcf2.VCF reader.

        Raises:
            VCFParsingError: If the file cannot be opened.
        """
        try:
            vcf = cyvcf2.VCF(self._vcf_path)
        except Exception as exc:
            raise VCFParsingError(
                f"Cannot open VCF file '{self._vcf_path}': {exc}"
            ) from exc
        self._csq_fields = self._parse_csq_header(vcf)
        return vcf

    def _parse_csq_header(self, vcf: cyvcf2.VCF) -> list[str]:
        """Extract CSQ sub-field names from the VCF header.

        Args:
            vcf: An open cyvcf2.VCF reader.

        Returns:
            Ordered list of CSQ sub-field names.
        """
        for header_line in vcf.header_iter():
            info = header_line.info(extra=True)
            if info.get("ID") == "CSQ":
                description = info.get("Description", "")
                match = re.search(r"Format: (.+?)(?:\"|$)", description)
                if match:
                    fields = match.group(1).strip().split("|")
                    self._logger.debug("Parsed %d CSQ fields from header", len(fields))
                    return fields
        self._logger.warning("CSQ header not found; using default field order")
        return _DEFAULT_CSQ_FIELDS

    def _field_index(self, name: str) -> int | None:
        """Return zero-based index of a CSQ sub-field, or None if absent.

        Args:
            name: CSQ sub-field name.

        Returns:
            Zero-based index, or None.
        """
        try:
            return self._csq_fields.index(name)
        except ValueError:
            return None

    def _parse_csq_entry(self, entry: str) -> dict[str, str]:
        """Parse a single pipe-delimited CSQ entry into a field mapping.

        Args:
            entry: A single CSQ annotation string (pipe-delimited).

        Returns:
            Dict mapping field name to value.
        """
        parts = entry.split("|")
        return {
            field: (parts[i] if i < len(parts) else "")
            for i, field in enumerate(self._csq_fields)
        }

    def _extract_vaf(self, variant: cyvcf2.Variant) -> float:
        """Extract tumour variant allele fraction from a VCF record.

        Reads from the tumour sample (``self._tumor_sample_index``).
        Tries FORMAT.AF first, then falls back to FORMAT.AD (allelic depths).

        Args:
            variant: A cyvcf2 variant record.

        Returns:
            VAF as a float, or 0.0 if not computable.
        """
        idx = self._tumor_sample_index
        try:
            af = variant.format("AF")
            if af is not None and len(af) > idx:
                value = af[idx]
                if hasattr(value, "__len__"):
                    value = value[0]
                if value is not None and float(value) >= 0:
                    return float(value)
        except Exception:
            pass
        try:
            ad = variant.format("AD")
            if ad is not None and len(ad) > idx:
                depths = ad[idx]
                if depths is not None and len(depths) >= 2 and sum(depths) > 0:
                    return float(depths[1]) / float(sum(depths))
        except Exception:
            pass
        return 0.0

    def _extract_expression(
        self,
        variant: cyvcf2.Variant,
        transcript_id: str,
    ) -> float | None:
        """Extract tumour RNA expression for a transcript from FORMAT fields.

        Checks FORMAT.TX (transcript-level TPM) first, then falls back to
        FORMAT.GX (gene-level TPM). Both fields use the format
        ``ID|TPM[,ID|TPM,...]`` and are populated by pVACtools-style pipelines.

        Transcript version numbers (e.g. the ``.5`` in ``ENST00000332831.5``)
        are stripped before matching so that versioned and unversioned IDs
        compare correctly.

        Args:
            variant: A cyvcf2 variant record.
            transcript_id: Ensembl transcript ID to look up in FORMAT.TX.

        Returns:
            Expression level in TPM, or None if not available.
        """
        idx = self._tumor_sample_index
        t_base = transcript_id.split(".")[0]

        # Transcript-level expression (preferred)
        try:
            tx = variant.format("TX")
            if tx is not None and len(tx) > idx:
                tx_val = tx[idx]
                if hasattr(tx_val, "__iter__") and not isinstance(tx_val, (str, bytes)):
                    tx_val = tx_val[0]
                tx_str = str(tx_val) if tx_val is not None else ""
                if tx_str and tx_str not in (".", ""):
                    for entry in tx_str.split(","):
                        if "|" in entry:
                            tid, tpm = entry.rsplit("|", 1)
                            if tid.split(".")[0] == t_base:
                                return float(tpm)
        except Exception:
            pass

        # Gene-level expression (fallback)
        try:
            gx = variant.format("GX")
            if gx is not None and len(gx) > idx:
                gx_val = gx[idx]
                if hasattr(gx_val, "__iter__") and not isinstance(gx_val, (str, bytes)):
                    gx_val = gx_val[0]
                gx_str = str(gx_val) if gx_val is not None else ""
                if gx_str and gx_str not in (".", "") and "|" in gx_str:
                    _, tpm = gx_str.rsplit("|", 1)
                    return float(tpm)
        except Exception:
            pass

        return None

    # ── Consequence matching ─────────────────────────────────────────────────

    def _match_consequence(
        self,
        consequence: str,
        requested: set[str],
    ) -> str | None:
        """Return the first term in *requested* that appears in a VEP consequence string.

        VEP can report compound consequences (ampersand-separated), e.g.
        "missense_variant&splice_region_variant".

        Args:
            consequence: Full VEP consequence string.
            requested: Consequence terms to match against.

        Returns:
            The matched term, or None.
        """
        for term in consequence.split("&"):
            if term in requested:
                return term
        return None

    # ── HGVSp parsing ────────────────────────────────────────────────────────

    def _parse_hgvsp(self, hgvsp: str) -> _ParsedHGVSp | None:
        """Parse an HGVSp string into amino acid change components.

        Handles missense substitutions, inframe deletions (single and range),
        inframe insertions, single-residue duplications, and frameshifts.

        **Important**: the ``variant_type`` and ``is_frameshift`` fields on the
        returned object are heuristic inferences from the HGVSp notation alone.
        Some frameshifts produce HGVSp strings that are structurally
        indistinguishable from missense notation (e.g. ``p.Glu11Ter`` — an
        immediate stop that matches the missense regex).  Callers that have
        access to the VEP ``Consequence`` field **must** override these two
        fields using :data:`_CONSEQUENCE_TO_VTYPE` and
        :data:`CONSEQUENCE_FRAMESHIFT` rather than trusting the values here.

        Args:
            hgvsp: HGVSp string, optionally prefixed with a transcript ID
                (e.g. "ENSP00000288602.7:p.Val600Glu").

        Returns:
            A _ParsedHGVSp instance, or None if the string cannot be parsed.
        """
        if ":" in hgvsp:
            hgvsp = hgvsp.split(":", 1)[1]
        if not hgvsp.startswith("p."):
            return None

        # Frameshift: p.Val600Glyfs*12 / p.Val600fs
        m = _RE_FRAMESHIFT.match(hgvsp)
        if m:
            aa_ref = _three_to_one(m.group(1))
            aa_pos = int(m.group(2))
            aa_alt = _three_to_one(m.group(3)) if m.group(3) else ""
            return _ParsedHGVSp(
                aa_ref=aa_ref, aa_alt=aa_alt, aa_pos=aa_pos,
                variant_type="frameshift", is_frameshift=True,
            )

        # Inframe range deletion: p.Val600_Lys601del
        m = _RE_DEL_RANGE.match(hgvsp)
        if m:
            aa_ref = _three_to_one(m.group(1)) + _three_to_one(m.group(3))
            aa_pos = int(m.group(2))
            return _ParsedHGVSp(
                aa_ref=aa_ref, aa_alt="", aa_pos=aa_pos,
                variant_type="inframe_deletion",
            )

        # Inframe single deletion: p.Val600del
        m = _RE_DEL_SINGLE.match(hgvsp)
        if m:
            return _ParsedHGVSp(
                aa_ref=_three_to_one(m.group(1)), aa_alt="",
                aa_pos=int(m.group(2)), variant_type="inframe_deletion",
            )

        # Inframe insertion: p.Ala600_Ala601insGlyPro
        m = _RE_INS.match(hgvsp)
        if m:
            aa_alt = _parse_aa_sequence(m.group(5))
            return _ParsedHGVSp(
                aa_ref="", aa_alt=aa_alt, aa_pos=int(m.group(2)),
                variant_type="inframe_insertion",
            )

        # Duplication (single residue): p.Val600dup → inframe insertion
        m = _RE_DUP_SINGLE.match(hgvsp)
        if m:
            aa = _three_to_one(m.group(1))
            return _ParsedHGVSp(
                aa_ref="", aa_alt=aa, aa_pos=int(m.group(2)),
                variant_type="inframe_insertion",
            )

        # Missense: p.Val600Glu
        m = _RE_MISSENSE.match(hgvsp)
        if m:
            return _ParsedHGVSp(
                aa_ref=_three_to_one(m.group(1)),
                aa_alt=_three_to_one(m.group(3)),
                aa_pos=int(m.group(2)),
                variant_type="missense",
            )

        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def read_variants(
        self,
        consequences: set[str] | None = None,
    ) -> list[SomaticVariant]:
        """Read variants matching the specified VEP consequence types.

        Args:
            consequences: Set of VEP consequence terms to include. Defaults
                to all types in SUPPORTED_CONSEQUENCES when None.

        Returns:
            List of SomaticVariant instances, one per matching transcript
            annotation.

        Raises:
            VCFParsingError: If the file is unreadable or critically malformed.
        """
        if consequences is None:
            consequences = set(SUPPORTED_CONSEQUENCES)

        vcf = self._open_vcf()
        variants: list[SomaticVariant] = []
        n_records = 0
        n_skipped = 0

        try:
            for record in vcf:
                n_records += 1
                csq_raw = record.INFO.get("CSQ")
                if csq_raw is None:
                    n_skipped += 1
                    continue

                csq_entries = (
                    csq_raw.split(",") if isinstance(csq_raw, str) else list(csq_raw)
                )
                vaf = self._extract_vaf(record)

                for entry in csq_entries:
                    parsed = self._parse_csq_entry(entry)
                    matched = self._match_consequence(
                        parsed.get("Consequence", ""), consequences
                    )
                    if matched is None:
                        continue

                    hgvsp = parsed.get("HGVSp", "")
                    if not hgvsp:
                        continue

                    hgvsp_result = self._parse_hgvsp(hgvsp)
                    if hgvsp_result is None:
                        self._logger.debug(
                            "Cannot parse HGVSp '%s' at %s:%d",
                            hgvsp, record.CHROM, record.POS,
                        )
                        continue

                    protein_change = hgvsp if hgvsp.startswith("p.") else f"p.{hgvsp}"
                    if ":" in protein_change:
                        protein_change = protein_change.split(":", 1)[1]

                    gene = parsed.get("SYMBOL", "") or parsed.get("Gene", "")
                    transcript_id = parsed.get("Feature", "")
                    if not gene or not transcript_id:
                        continue

                    expression = self._extract_expression(record, transcript_id)

                    # Derive classification from the VEP Consequence field —
                    # the authoritative source.  The HGVSp string alone is
                    # insufficient: e.g. p.Glu11Ter (an immediate stop codon
                    # from a frameshift) is structurally identical to a
                    # stop-gained missense and would be misclassified if we
                    # relied on _parse_hgvsp's heuristic variant_type.
                    is_frameshift: bool = matched == CONSEQUENCE_FRAMESHIFT
                    variant_type: str = _CONSEQUENCE_TO_VTYPE[matched]

                    downstream_sequence: str | None = None
                    if is_frameshift:
                        ds = parsed.get("FrameshiftSequence", "")
                        if ds:
                            downstream_sequence = ds

                    wt = parsed.get("WildtypeProtein", "") or None

                    variants.append(SomaticVariant(
                        chrom=record.CHROM,
                        pos=record.POS,
                        ref=record.REF,
                        alt=record.ALT[0] if record.ALT else "",
                        gene=gene,
                        transcript_id=transcript_id,
                        protein_change=protein_change,
                        aa_ref=hgvsp_result.aa_ref,
                        aa_alt=hgvsp_result.aa_alt,
                        aa_pos=hgvsp_result.aa_pos,
                        vaf=vaf,
                        expression=expression,
                        consequence=matched,
                        variant_type=variant_type,
                        is_frameshift=is_frameshift,
                        downstream_sequence=downstream_sequence,
                        wildtype_protein_sequence=wt,
                    ))
        except VCFParsingError:
            raise
        except Exception as exc:
            raise VCFParsingError(
                f"Error reading VCF '{self._vcf_path}': {exc}"
            ) from exc
        finally:
            vcf.close()

        self._logger.info(
            "Parsed %d records; extracted %d variants of types %s (%d skipped)",
            n_records, len(variants), sorted(consequences), n_skipped,
        )
        return variants

    def read_missense_variants(self) -> list[SomaticVariant]:
        """Convenience method: read missense variants only.

        Returns:
            List of missense SomaticVariant instances.
        """
        return self.read_variants({CONSEQUENCE_MISSENSE})
