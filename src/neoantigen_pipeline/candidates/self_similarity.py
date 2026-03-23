"""Self-similarity filtering against the reference proteome.

Computes the degree of similarity between neoantigen peptide candidates
and the host self-proteome using sliding-window identity scoring.
High self-similarity indicates the peptide resembles a self-peptide and
may not elicit an immune response.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from neoantigen_pipeline._constants import DEFAULT_SIMILARITY_THRESHOLD

if TYPE_CHECKING:
    from neoantigen_pipeline.io.proteome import ProteomeDB


class SelfSimilarityFilter:
    """Computes self-similarity of peptide candidates against the reference proteome.

    Uses a sliding-window approach to find the maximum sequence identity
    between the query peptide and any window of equal length in the proteome.
    This is O(n * m) where n = total proteome length and m = peptide length,
    but remains practical for single peptide lookups.

    For large-scale use, replace with a BLAST-based approach.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(type(self).__qualname__)

    def compute_similarity(self, peptide: str, proteome_db: ProteomeDB) -> float:
        """Compute the maximum fractional identity to any proteome window.

        Scans every protein sequence in the proteome database with a window
        of the same length as the query peptide and returns the highest
        fractional identity found.

        Args:
            peptide: Query peptide amino acid sequence.
            proteome_db: Reference proteome database.

        Returns:
            Maximum fractional identity in [0.0, 1.0].
            Returns 0.0 if the proteome is empty or peptide is longer than
            all sequences.
        """
        if not peptide:
            return 0.0

        k = len(peptide)
        max_identity: float = 0.0

        for seq in proteome_db.iter_sequences():
            if len(seq) < k:
                continue
            identity = self._max_window_identity(peptide, seq, k)
            if identity > max_identity:
                max_identity = identity
                if max_identity == 1.0:
                    return 1.0  # Early exit on perfect match

        return max_identity

    def _max_window_identity(self, peptide: str, sequence: str, k: int) -> float:
        """Find maximum per-position identity in a sliding window.

        Args:
            peptide: Query peptide of length k.
            sequence: Target protein sequence.
            k: Window length.

        Returns:
            Maximum fractional identity across all windows.
        """
        best: float = 0.0
        for i in range(len(sequence) - k + 1):
            matches = sum(a == b for a, b in zip(peptide, sequence[i : i + k]))
            identity = matches / k
            if identity > best:
                best = identity
        return best

    def is_self_similar(
        self,
        peptide: str,
        proteome_db: ProteomeDB,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> bool:
        """Determine whether a peptide is sufficiently similar to self-proteome.

        Args:
            peptide: Query peptide amino acid sequence.
            proteome_db: Reference proteome database.
            threshold: Fractional identity threshold above which a peptide is
                considered self-similar (default ``DEFAULT_SIMILARITY_THRESHOLD``).

        Returns:
            True if the peptide's maximum self-similarity exceeds the threshold.
        """
        similarity = self.compute_similarity(peptide, proteome_db)
        result = similarity >= threshold
        self._logger.debug(
            "Peptide '%s': self-similarity=%.3f, threshold=%.3f, self_similar=%s",
            peptide,
            similarity,
            threshold,
            result,
        )
        return result
