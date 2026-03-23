"""Abstract base class for MHC binding predictors.

Defines the interface that all concrete predictor implementations must satisfy,
enabling dependency injection and easy substitution of prediction backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neoantigen_pipeline.prediction.results import BindingPrediction


class BindingPredictor(ABC):
    """Abstract base class for MHC peptide binding predictors.

    Concrete subclasses (e.g. ``MHCflurryPredictor``) implement the abstract
    methods for a specific MHC class and prediction backend.

    Subclass responsibility:
        - Override ``predict`` to return typed ``BindingPrediction`` results.
        - Set ``name`` to a human-readable backend identifier.
        - Set ``mhc_class`` to 1 or 2.
    """

    @abstractmethod
    def predict(
        self, peptides: list[str], alleles: list[str]
    ) -> list[BindingPrediction]:
        """Predict MHC binding for a list of peptides and alleles.

        Args:
            peptides: List of amino acid sequence strings.
            alleles: List of HLA allele strings in "HLA-A*02:01" notation.

        Returns:
            List of ``BindingPrediction`` objects, one per input peptide,
            containing at minimum the peptide sequence, best allele, and
            affinity/presentation scores.

        Raises:
            PredictionError: If prediction fails.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this predictor backend.

        Returns:
            Predictor name string, e.g. "MHCflurry2-Class1Presentation".
        """

    @property
    @abstractmethod
    def mhc_class(self) -> int:
        """MHC restriction class.

        Returns:
            1 for MHC class I, 2 for MHC class II.
        """
