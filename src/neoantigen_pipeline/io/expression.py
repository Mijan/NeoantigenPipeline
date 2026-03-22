"""Expression data loading from VCF annotations and TSV files.

Provides classes for loading and querying gene expression data (TPM/FPKM)
from multiple input formats.
"""

from __future__ import annotations

import logging
from pathlib import Path


class ExpressionData:
    """Container for gene-level expression data.

    Wraps a dictionary mapping gene symbol (or Ensembl ID) to expression
    value in TPM or FPKM units.

    Args:
        expression_map: Dict of gene identifier -> expression value.
    """

    def __init__(self, expression_map: dict[str, float]) -> None:
        self._data: dict[str, float] = dict(expression_map)
        self._logger = logging.getLogger(type(self).__qualname__)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def genes(self) -> frozenset[str]:
        """Set of gene identifiers in this expression dataset.

        Returns:
            Frozenset of gene identifier strings.
        """
        return frozenset(self._data)

    def get(self, gene: str) -> float | None:
        """Look up expression value for a gene.

        Args:
            gene: Gene symbol or Ensembl ID.

        Returns:
            Expression value, or None if the gene is not present.
        """
        return self._data.get(gene)

    def get_or_default(self, gene: str, default: float = 0.0) -> float:
        """Look up expression value with a fallback default.

        Args:
            gene: Gene symbol or Ensembl ID.
            default: Value to return if gene is absent.

        Returns:
            Expression value, or the default.
        """
        return self._data.get(gene, default)


class ExpressionLoader:
    """Loads gene expression data from various file formats.

    Supports two input modes:
    1. Extraction from VEP-annotated VCF INFO/CSQ fields.
    2. Loading from a tab-separated TSV file (gene, expression columns).
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(type(self).__qualname__)

    def load_from_vcf(self, vcf_path: str) -> ExpressionData:
        """Extract per-gene expression values from VEP CSQ annotations in a VCF.

        The method re-uses VCFReader internals to scan CSQ fields for any
        sub-field named "EXPRESSION" or "TPM" and builds a gene->expression map.

        Args:
            vcf_path: Path to the VEP-annotated VCF file.

        Returns:
            ExpressionData populated from VCF annotations.
        """

        from neoantigen_pipeline.io.vcf_reader import VCFReader

        reader = VCFReader(vcf_path)
        vcf = reader._open_vcf()  # noqa: SLF001 – intentional reuse
        expression_map: dict[str, float] = {}

        try:
            for record in vcf:
                csq_raw = record.INFO.get("CSQ")
                if csq_raw is None:
                    continue
                if isinstance(csq_raw, str):
                    entries = csq_raw.split(",")
                else:
                    entries = list(csq_raw)

                for entry in entries:
                    parsed = reader._parse_csq_entry(entry)  # noqa: SLF001
                    gene = parsed.get("SYMBOL", "") or parsed.get("Gene", "")
                    if not gene:
                        continue
                    for key in ("EXPRESSION", "expression", "TPM", "tpm"):
                        val_str = parsed.get(key, "")
                        if val_str:
                            try:
                                expression_map[gene] = float(val_str)
                                break
                            except ValueError:
                                pass
        finally:
            vcf.close()

        self._logger.info(
            "Extracted expression data for %d genes from '%s'",
            len(expression_map),
            vcf_path,
        )
        return ExpressionData(expression_map)

    def load_from_tsv(self, path: str) -> ExpressionData:
        """Load expression data from a two-column (or more) TSV file.

        Expected format: tab-separated with a header row. Recognises columns
        named "gene", "gene_id", "gene_name" (first), and "tpm", "fpkm",
        "expression", "count" (second numeric column).

        Args:
            path: Path to the TSV expression file.

        Returns:
            ExpressionData populated from the TSV.

        Raises:
            OSError: If the file cannot be opened.
            ValueError: If expected columns are absent.
        """
        expression_map: dict[str, float] = {}
        tsv_path = Path(path)

        with tsv_path.open(encoding="utf-8") as fh:
            header_line = fh.readline().rstrip("\n")
            header = [h.lower().strip() for h in header_line.split("\t")]

            # Identify gene and expression columns
            gene_col = next(
                (
                    i
                    for i, h in enumerate(header)
                    if h in ("gene", "gene_id", "gene_name")
                ),
                0,
            )
            expr_col = next(
                (
                    i
                    for i, h in enumerate(header)
                    if h in ("tpm", "fpkm", "expression", "count", "expected_count")
                ),
                1,
            )

            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(gene_col, expr_col):
                    continue
                gene = parts[gene_col].strip()
                if not gene:
                    continue
                try:
                    expression_map[gene] = float(parts[expr_col].strip())
                except ValueError:
                    pass

        self._logger.info(
            "Loaded expression data for %d genes from '%s'", len(expression_map), path
        )
        return ExpressionData(expression_map)
