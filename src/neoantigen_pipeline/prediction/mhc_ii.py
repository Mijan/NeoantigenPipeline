"""MHC class II binding predictor placeholder.

MHC class II prediction will be implemented in a future release using a
transformer-based model. This module provides the class skeleton that
satisfies the BindingPredictor interface so the pipeline can be extended
without structural changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from neoantigen_pipeline.prediction.base import BindingPredictor

if TYPE_CHECKING:
    pass


class MHCIIPredictor(BindingPredictor):
    """Placeholder MHC class II binding predictor.

    This class satisfies the BindingPredictor abstract interface but raises
    NotImplementedError on any prediction call. It will be replaced by a
    transformer-based HLA-II prediction model in a future release.

    Args:
        alleles: HLA-II allele strings (e.g. "HLA-DRB1*01:01").
    """

    def __init__(self, alleles: tuple[str, ...] = ()) -> None:
        self._alleles = alleles
        self._logger = logging.getLogger(type(self).__qualname__)

    @property
    def name(self) -> str:
        """Predictor name.

        Returns:
            "MHCIIPredictor-Placeholder"
        """
        return "MHCIIPredictor-Placeholder"

    @property
    def mhc_class(self) -> int:
        """MHC class.

        Returns:
            2 (class II)
        """
        return 2

    def predict(self, peptides: list[str], alleles: list[str]) -> pd.DataFrame:
        """Not implemented.

        Args:
            peptides: List of amino acid sequences.
            alleles: List of HLA-II allele strings.

        Raises:
            NotImplementedError: Always. MHC-II prediction is not yet available.
        """
        raise NotImplementedError(
            "MHC class II prediction is not yet implemented. "
            "This feature is planned for a future release."
        )
