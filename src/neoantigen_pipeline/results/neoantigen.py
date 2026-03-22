"""Neoantigen result data containers.

Provides immutable dataclasses for individual neoantigen candidates and
a result set container with serialisation to CSV and Pandas DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class NeoantigenCandidate:
    """An individual neoantigen candidate with all prediction scores.

    This is the primary result object produced by the pipeline. All fields
    are populated by the ranking step and should not be mutated after creation.

    Attributes:
        gene: HGNC gene symbol.
        mutation: Human-readable mutation string, e.g. "BRAF_p.Val600Glu".
        peptide: Mutant k-mer peptide sequence.
        wildtype_peptide: Corresponding wildtype peptide at the same position.
        best_allele: HLA allele with the highest presentation score.
        presentation_score: MHCflurry presentation score (0–1).
        binding_affinity_nm: Mutant peptide IC50 binding affinity in nM.
        wildtype_affinity_nm: Wildtype peptide IC50 binding affinity in nM.
        processing_score: Antigen processing score from MHCflurry.
        agretopicity: Ratio of wildtype to mutant binding affinity.
        expression: Gene expression level in TPM/FPKM.
        vaf: Variant allele fraction (0–1).
        composite_score: Weighted composite neoantigen score.
        composite_rank: 1-based rank among all candidates (lower = better).
    """

    __slots__ = (
        "gene",
        "mutation",
        "peptide",
        "wildtype_peptide",
        "best_allele",
        "presentation_score",
        "binding_affinity_nm",
        "wildtype_affinity_nm",
        "processing_score",
        "agretopicity",
        "expression",
        "vaf",
        "composite_score",
        "composite_rank",
    )

    gene: str
    mutation: str
    peptide: str
    wildtype_peptide: str
    best_allele: str
    presentation_score: float
    binding_affinity_nm: float
    wildtype_affinity_nm: float
    processing_score: float
    agretopicity: float
    expression: float
    vaf: float
    composite_score: float
    composite_rank: int


class NeoantigenResultSet:
    """Container for a collection of ranked neoantigen candidates.

    Wraps a list of NeoantigenCandidate objects and provides DataFrame and
    CSV serialisation.

    Args:
        candidates: List of NeoantigenCandidate instances, typically
            pre-sorted by composite_rank.
    """

    def __init__(self, candidates: list[NeoantigenCandidate]) -> None:
        self._candidates = list(candidates)
        self._logger = logging.getLogger(type(self).__qualname__)

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self):
        return iter(self._candidates)

    def __repr__(self) -> str:
        return f"NeoantigenResultSet(n={len(self._candidates)})"

    @property
    def candidates(self) -> list[NeoantigenCandidate]:
        """Ordered list of neoantigen candidates.

        Returns:
            List of NeoantigenCandidate instances.
        """
        return list(self._candidates)

    @property
    def top_candidate(self) -> NeoantigenCandidate | None:
        """The highest-ranked candidate (rank 1), or None if empty.

        Returns:
            NeoantigenCandidate with composite_rank=1, or None.
        """
        if not self._candidates:
            return None
        return min(self._candidates, key=lambda c: c.composite_rank)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the result set to a Pandas DataFrame.

        Returns:
            DataFrame where each row corresponds to one NeoantigenCandidate.
            Column names match NeoantigenCandidate field names.
        """
        if not self._candidates:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=list(NeoantigenCandidate.__slots__))

        return pd.DataFrame(
            [
                {slot: getattr(c, slot) for slot in NeoantigenCandidate.__slots__}
                for c in self._candidates
            ]
        )

    def to_csv(self, path: str) -> None:
        """Write the result set to a CSV file.

        The output directory is created if it does not exist.

        Args:
            path: Destination file path (will be created or overwritten).
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        self._logger.info(
            "Wrote %d neoantigen candidates to '%s'", len(self._candidates), path
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> NeoantigenResultSet:
        """Reconstruct a NeoantigenResultSet from a DataFrame.

        The DataFrame must have columns matching all NeoantigenCandidate fields.
        This is typically used to reload previously written CSV output.

        Args:
            df: DataFrame with NeoantigenCandidate field columns.

        Returns:
            NeoantigenResultSet populated from the DataFrame rows.

        Raises:
            KeyError: If required columns are absent.
        """
        required = set(NeoantigenCandidate.__slots__)
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"DataFrame is missing required columns: {missing}"
            )

        candidates: list[NeoantigenCandidate] = []
        for _, row in df.iterrows():
            candidates.append(
                NeoantigenCandidate(
                    gene=str(row["gene"]),
                    mutation=str(row["mutation"]),
                    peptide=str(row["peptide"]),
                    wildtype_peptide=str(row["wildtype_peptide"]),
                    best_allele=str(row["best_allele"]),
                    presentation_score=float(row["presentation_score"]),
                    binding_affinity_nm=float(row["binding_affinity_nm"]),
                    wildtype_affinity_nm=float(row["wildtype_affinity_nm"]),
                    processing_score=float(row["processing_score"]),
                    agretopicity=float(row["agretopicity"]),
                    expression=float(row["expression"]),
                    vaf=float(row["vaf"]),
                    composite_score=float(row["composite_score"]),
                    composite_rank=int(row["composite_rank"]),
                )
            )

        logger = logging.getLogger(cls.__qualname__)
        logger.info("Loaded %d candidates from DataFrame", len(candidates))
        return cls(candidates)
