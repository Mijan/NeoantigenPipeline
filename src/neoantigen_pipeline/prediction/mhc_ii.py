"""Abstract MHC class II binding predictor interface.

Placeholder for a future transformer-based HLA-II prediction model.
Provides the abstract interface that concrete implementations must satisfy.
"""

from __future__ import annotations

from neoantigen_pipeline.prediction.base import BindingPredictor
from neoantigen_pipeline.prediction.results import BindingPrediction


class MHCIIPredictor(BindingPredictor):
    """Abstract base class for MHC class II binding predictors.

    MHC class II prediction will be implemented in a future release using a
    transformer-based model. This class defines the interface that all
    concrete HLA-II backends must satisfy.

    Concrete subclasses must implement ``predict`` and ``name``.
    """

    @property
    def mhc_class(self) -> int:
        """MHC class.

        Returns:
            2 (class II)
        """
        return 2

    def predict(self, peptides: list[str], alleles: list[str]) -> list[BindingPrediction]:
        """Not implemented.

        Args:
            peptides: List of amino acid sequences.
            alleles: List of HLA-II allele strings.

        Raises:
            NotImplementedError: Always — MHC-II prediction is not yet available.
        """
        raise NotImplementedError(
            "MHC class II prediction is not yet implemented. "
            "This feature is planned for a future release using a "
            "transformer-based model."
        )

    @property
    def name(self) -> str:
        """Predictor name.

        Returns:
            "MHCIIPredictor-Placeholder"
        """
        return "MHCIIPredictor-Placeholder"
