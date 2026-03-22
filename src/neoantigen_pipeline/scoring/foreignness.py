"""Foreignness scoring for neoantigen candidates.

Foreignness quantifies how dissimilar a neoantigen peptide is from the
self-proteome. Higher foreignness implies greater immunogenic potential
since the immune system is less likely to be tolerant to the peptide.
"""

from __future__ import annotations

import logging


class ForeignnessScorer:
    """Computes foreignness scores for neoantigen peptide candidates.

    Foreignness is defined as the complement of self-similarity:
    ``foreignness = 1.0 - similarity_score``

    A score of 1.0 indicates complete foreignness (no resemblance to any
    self-peptide), while 0.0 indicates perfect self-identity.

    For a richer foreignness model, the similarity score can be derived from
    BLOSUM62-weighted alignment scores rather than simple percent identity.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(type(self).__qualname__)

    def compute(self, similarity_score: float) -> float:
        """Compute foreignness from a self-similarity score.

        Args:
            similarity_score: Fractional identity to the closest self-peptide,
                in [0.0, 1.0]. Obtained from SelfSimilarityFilter.compute_similarity.

        Returns:
            Foreignness score in [0.0, 1.0].

        Raises:
            ValueError: If similarity_score is outside [0.0, 1.0].
        """
        if not (0.0 <= similarity_score <= 1.0):
            raise ValueError(
                f"similarity_score must be in [0, 1]; got {similarity_score}"
            )
        return 1.0 - similarity_score

    def compute_batch(self, similarity_scores: list[float]) -> list[float]:
        """Compute foreignness for a list of similarity scores.

        Args:
            similarity_scores: List of fractional identity values.

        Returns:
            List of foreignness scores.
        """
        return [self.compute(s) for s in similarity_scores]
