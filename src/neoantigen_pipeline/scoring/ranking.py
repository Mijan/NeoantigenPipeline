"""Composite neoantigen ranking.

Combines multiple evidence streams (MHC presentation, agretopicity, expression,
VAF) into a single composite score and produces a ranked list of
``NeoantigenCandidate`` objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from neoantigen_pipeline.results.neoantigen import NeoantigenCandidate

if TYPE_CHECKING:
    from neoantigen_pipeline.config import ScoringConfig
    from neoantigen_pipeline.prediction.results import ScoredCandidate


class RankingScorer:
    """Ranks neoantigen candidates by a weighted composite score.

    Each component score is normalised to [0, 1] using min-max scaling so
    that all features contribute on the same scale. The composite score is the
    weighted sum of normalised components, with weights from ``ScoringConfig``.

    Args:
        config: Scoring configuration specifying component weights.
    """

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(type(self).__qualname__)

    def rank(self, candidates: list[ScoredCandidate]) -> list[NeoantigenCandidate]:
        """Compute composite scores and return candidates sorted by rank.

        Extracts per-component score arrays, normalises each to [0, 1] via
        min-max scaling, computes the weighted composite, sorts descending,
        and assigns 1-based ranks.

        Args:
            candidates: Scored candidates to rank. Must be non-empty.

        Returns:
            List of ``NeoantigenCandidate`` objects sorted by composite score
            (highest first), with ``composite_rank`` set to 1, 2, 3 …

        Raises:
            ValueError: If ``candidates`` is empty.
        """
        if not candidates:
            raise ValueError("Cannot rank an empty candidate list")

        presentation = np.array([c.presentation_score for c in candidates])
        agretopicity = np.array([c.agretopicity for c in candidates])
        expression = np.array([c.expression for c in candidates])
        vaf = np.array([c.vaf for c in candidates])

        composite = (
            self._config.presentation_score_weight * self._normalise(presentation)
            + self._config.agretopicity_weight * self._normalise(agretopicity)
            + self._config.expression_weight * self._normalise(expression)
            + self._config.vaf_weight * self._normalise(vaf)
        )

        order = np.argsort(-composite)  # descending

        ranked: list[NeoantigenCandidate] = []
        for rank, idx in enumerate(order, start=1):
            c = candidates[int(idx)]
            ranked.append(
                NeoantigenCandidate(
                    gene=c.gene,
                    mutation=c.mutation_str,
                    peptide=c.peptide,
                    wildtype_peptide=c.wildtype_peptide,
                    best_allele=c.best_allele,
                    presentation_score=c.presentation_score,
                    binding_affinity_nm=c.binding_affinity_nm,
                    wildtype_affinity_nm=c.wildtype_affinity_nm,
                    processing_score=c.processing_score,
                    agretopicity=c.agretopicity,
                    expression=c.expression,
                    vaf=c.vaf,
                    composite_score=float(composite[idx]),
                    composite_rank=rank,
                )
            )

        self._logger.info("Ranked %d neoantigen candidates", len(ranked))
        return ranked

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        """Min-max normalise an array to [0, 1].

        If all values are identical (zero range), returns a zero array.

        Args:
            values: 1-D NumPy array of numeric values.

        Returns:
            Normalised array of the same length.
        """
        min_val = values.min()
        value_range = values.max() - min_val
        if value_range == 0.0:
            return np.zeros_like(values)
        return (values - min_val) / value_range
