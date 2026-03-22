"""Expression-based variant filtering.

Filters somatic variants by their associated gene expression level,
removing lowly or unexpressed genes before peptide generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neoantigen_pipeline.config import ExpressionFilterConfig
    from neoantigen_pipeline.io.vcf_reader import SomaticVariant


class ExpressionFilter:
    """Filters somatic variants based on gene expression thresholds.

    Variants whose expression exceeds `min_expression` are retained.
    Variants with no expression annotation are kept or removed depending
    on the `filter_missing` configuration flag.

    Args:
        config: Expression filter configuration.
    """

    def __init__(self, config: ExpressionFilterConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(type(self).__qualname__)

    def filter(self, variants: list[SomaticVariant]) -> list[SomaticVariant]:
        """Filter a list of somatic variants by expression level.

        Args:
            variants: Input list of somatic variants.

        Returns:
            Filtered list containing only variants that pass the expression
            threshold. If a variant has no expression annotation, it is
            kept (filter_missing=False) or removed (filter_missing=True).
        """
        if not variants:
            return []

        retained: list[SomaticVariant] = []
        n_no_expression = 0
        n_below_threshold = 0

        for variant in variants:
            if variant.expression is None:
                n_no_expression += 1
                if not self._config.filter_missing:
                    retained.append(variant)
                else:
                    self._logger.debug(
                        "Removing variant %s (no expression data)",
                        variant.protein_change,
                    )
            elif variant.expression < self._config.min_expression:
                n_below_threshold += 1
                self._logger.debug(
                    "Removing variant %s: expression %.3f < threshold %.3f",
                    variant.protein_change,
                    variant.expression,
                    self._config.min_expression,
                )
            else:
                retained.append(variant)

        self._logger.info(
            "Expression filter: %d/%d variants retained "
            "(%d no-expression, %d below threshold %.3f)",
            len(retained),
            len(variants),
            n_no_expression,
            n_below_threshold,
            self._config.min_expression,
        )
        return retained
