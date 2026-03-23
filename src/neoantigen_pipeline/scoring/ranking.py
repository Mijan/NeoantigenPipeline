"""Composite neoantigen ranking.

Combines multiple evidence streams (MHC presentation, agretopicity, expression,
VAF) into a single composite score and produces a ranked candidate list.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neoantigen_pipeline._constants import (
    COL_AGRETOPICITY,
    COL_COMPOSITE_RANK,
    COL_COMPOSITE_SCORE,
    COL_EXPRESSION,
    COL_PRESENTATION_SCORE,
    COL_VAF,
)

if TYPE_CHECKING:
    from neoantigen_pipeline.config import ScoringConfig


class RankingScorer:
    """Ranks neoantigen candidates by a weighted composite score.

    Each component score is normalised to [0, 1] using min-max scaling
    before weighting so that all features contribute on the same scale.
    The weights are taken from the ScoringConfig.

    Args:
        config: Scoring configuration specifying component weights.
    """

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(type(self).__qualname__)

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute composite scores and rank neoantigen candidates.

        Each present scoring column is min-max normalised to [0, 1].
        Missing columns default to 0.0 with a warning. The composite score
        is the weighted sum of normalised components.

        Args:
            df: Input DataFrame with prediction and annotation columns.

        Returns:
            Copy of the input DataFrame with added columns:
            - ``composite_score``: weighted sum of normalised component scores
            - ``composite_rank``: 1-based rank sorted by composite_score (desc)

        Raises:
            ValueError: If the DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot rank an empty DataFrame")

        df = df.copy()

        composite = (
            self._config.presentation_score_weight * self._normalise(df, COL_PRESENTATION_SCORE)
            + self._config.agretopicity_weight * self._normalise(df, COL_AGRETOPICITY)
            + self._config.expression_weight * self._normalise(df, COL_EXPRESSION)
            + self._config.vaf_weight * self._normalise(df, COL_VAF)
        )

        df[COL_COMPOSITE_SCORE] = composite
        df = df.sort_values(COL_COMPOSITE_SCORE, ascending=False).reset_index(drop=True)
        df[COL_COMPOSITE_RANK] = range(1, len(df) + 1)

        self._logger.info("Ranked %d neoantigen candidates", len(df))
        return df

    def _normalise(self, df: pd.DataFrame, column: str) -> np.ndarray:
        """Min-max normalise a DataFrame column to [0, 1].

        If the column is absent, returns a zero array with a warning.
        If all values are identical (zero range), returns a zero array.

        Args:
            df: Input DataFrame.
            column: Column name to normalise.

        Returns:
            NumPy array of normalised values, same length as df.
        """
        if column not in df.columns:
            self._logger.warning(
                "Scoring column '%s' not found; contribution set to 0.0", column
            )
            return np.zeros(len(df))

        values = df[column].fillna(0.0).to_numpy(dtype=float)
        min_val = values.min()
        value_range = values.max() - min_val

        if value_range == 0.0:
            return np.zeros(len(df))

        return (values - min_val) / value_range
