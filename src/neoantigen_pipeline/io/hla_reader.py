"""HLA type parsing from OptiType output and manual specification.

Supports reading HLA alleles from OptiType TSV result files and from
manually specified allele lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from neoantigen_pipeline.exceptions import HLAParsingError


@dataclass(frozen=True)
class PatientHLA:
    """HLA allele set for a single patient.

    Attributes:
        class_i_alleles: Tuple of HLA class I allele strings in
            "HLA-A*29:02" notation.
        class_ii_alleles: Tuple of HLA class II allele strings (may be empty).
    """

    class_i_alleles: tuple[str, ...]
    class_ii_alleles: tuple[str, ...] = ()

    @property
    def all_alleles(self) -> tuple[str, ...]:
        """All alleles combined (class I + class II).

        Returns:
            Concatenated tuple of all allele strings.
        """
        return self.class_i_alleles + self.class_ii_alleles


# Regex that matches OptiType allele values like "A*29:02" or "29:02" or "A2902"
_OPTITYPE_ALLELE_RE = re.compile(
    r"^(?P<locus>[A-C])\*?(?P<group>\d{2}):?(?P<protein>\d{2})$",
    re.IGNORECASE,
)

# Standard HLA allele notation e.g. "HLA-A*02:01"
_HLA_ALLELE_RE = re.compile(
    r"^HLA-(?P<locus>[A-Z]+)\*(?P<group>\d+):(?P<protein>\d+)(?::\d+)*$"
)


def _normalise_allele(raw: str) -> str:
    """Convert a raw allele string to "HLA-X*NN:NN" notation.

    Args:
        raw: Raw allele value, e.g. "A*29:02", "A2902", or "HLA-A*29:02".

    Returns:
        Normalised allele string in "HLA-A*29:02" format.

    Raises:
        HLAParsingError: If the allele cannot be parsed.
    """
    raw = raw.strip()

    # Already in full HLA notation
    if _HLA_ALLELE_RE.match(raw):
        return raw

    # Attempt OptiType-style parsing
    match = _OPTITYPE_ALLELE_RE.match(raw)
    if match:
        locus = match.group("locus").upper()
        group = match.group("group")
        protein = match.group("protein")
        return f"HLA-{locus}*{group}:{protein}"

    raise HLAParsingError(
        f"Cannot parse allele '{raw}' — expected format like 'A*29:02' or 'HLA-A*29:02'"
    )


class HLAReader:
    """Reads patient HLA types from OptiType output files or manual input.

    Each instance is stateless; all methods can be called on a shared instance
    or used directly via classmethods.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(type(self).__qualname__)

    def read_optitype(self, path: str) -> PatientHLA:
        """Parse an OptiType result TSV file into a PatientHLA record.

        OptiType produces a single-row TSV with columns:
        (index), A1, A2, B1, B2, C1, C2, Reads, Objective

        Args:
            path: Path to the OptiType result TSV file.

        Returns:
            PatientHLA with class_i_alleles populated.

        Raises:
            HLAParsingError: If the file cannot be read or the allele columns
                are missing or unparseable.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                lines = [line.rstrip("\n") for line in fh if line.strip()]
        except OSError as exc:
            raise HLAParsingError(f"Cannot open OptiType file '{path}': {exc}") from exc

        if len(lines) < 2:
            raise HLAParsingError(
                f"OptiType file '{path}' must have a header row and at least one data row"
            )

        header = lines[0].split("\t")
        data = lines[1].split("\t")

        # OptiType header may start with an unnamed index column
        if header[0] == "" or header[0].isdigit():
            header = header[1:]
            data = data[1:]

        col_map = {name.strip(): i for i, name in enumerate(header)}

        class_i_loci = ["A1", "A2", "B1", "B2", "C1", "C2"]
        alleles: list[str] = []

        for locus in class_i_loci:
            idx = col_map.get(locus)
            if idx is None:
                raise HLAParsingError(
                    f"OptiType file '{path}' missing expected column '{locus}'. "
                    f"Found columns: {list(col_map)}"
                )
            raw_allele = data[idx].strip() if idx < len(data) else ""
            if not raw_allele:
                self._logger.warning("Empty allele for locus %s in '%s'", locus, path)
                continue
            try:
                alleles.append(_normalise_allele(raw_allele))
            except HLAParsingError as exc:
                self._logger.warning(
                    "Skipping unparseable allele '%s' for %s: %s",
                    raw_allele,
                    locus,
                    exc,
                )

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_alleles: list[str] = []
        for allele in alleles:
            if allele not in seen:
                seen.add(allele)
                unique_alleles.append(allele)

        self._logger.info(
            "Read %d unique HLA-I alleles from '%s': %s",
            len(unique_alleles),
            path,
            unique_alleles,
        )
        return PatientHLA(class_i_alleles=tuple(unique_alleles))

    @classmethod
    def read_manual(cls, alleles: list[str]) -> PatientHLA:
        """Create a PatientHLA from a manually provided list of allele strings.

        Alleles must be in "HLA-A*29:02" notation. Class I (loci A, B, C)
        and class II (loci DP, DQ, DR) are separated automatically.

        Args:
            alleles: List of allele strings.

        Returns:
            PatientHLA with class_i_alleles and class_ii_alleles populated.

        Raises:
            HLAParsingError: If any allele cannot be parsed.
        """
        logger = logging.getLogger(cls.__qualname__)
        class_i: list[str] = []
        class_ii: list[str] = []

        for raw in alleles:
            normalised = _normalise_allele(raw)
            # Loci A, B, C → class I; DP, DQ, DR → class II
            locus_match = re.match(r"HLA-([A-Z]+)\*", normalised)
            locus = locus_match.group(1) if locus_match else ""
            if locus in ("A", "B", "C"):
                class_i.append(normalised)
            else:
                class_ii.append(normalised)

        logger.info(
            "Manual HLA input: %d class-I, %d class-II alleles",
            len(class_i),
            len(class_ii),
        )
        return PatientHLA(
            class_i_alleles=tuple(class_i),
            class_ii_alleles=tuple(class_ii),
        )
