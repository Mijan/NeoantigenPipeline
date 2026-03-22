"""Abstract base classes and protocols for MHC binding predictors.

Defines the interfaces that all concrete predictor implementations must
satisfy, enabling dependency injection and easy swapping of prediction backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


class BindingPredictor(ABC):
    """Abstract base class for MHC peptide binding predictors.

    All concrete predictor classes (MHCflurry, NetMHCpan, etc.) must subclass
    this and implement the abstract methods and properties.

    Attributes for subclasses to define:
        _config: Predictor-specific configuration dataclass.
        _logger: Module-level logger.
    """

    @abstractmethod
    def predict(self, peptides: list[str], alleles: list[str]) -> pd.DataFrame:
        """Predict MHC binding for a list of peptides and alleles.

        Args:
            peptides: List of amino acid sequence strings.
            alleles: List of HLA allele strings in "HLA-A*02:01" notation.

        Returns:
            DataFrame with at minimum columns:
            - ``peptide``: input peptide sequence
            - ``allele``: HLA allele
            - One or more score columns (implementation-specific)

        Raises:
            PredictionError: If prediction fails.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this predictor.

        Returns:
            Predictor name string, e.g. "MHCflurry2".
        """

    @property
    @abstractmethod
    def mhc_class(self) -> int:
        """MHC restriction class (I or II).

        Returns:
            1 for class I, 2 for class II.
        """


@runtime_checkable
class PresentationPredictorProtocol(Protocol):
    """Structural typing protocol for predictors that support processing scores.

    Any object implementing this protocol can be used in pipeline steps that
    require processing-aware prediction (e.g. mhcflurry Class1PresentationPredictor).
    """

    def predict(
        self,
        peptides: list[str],
        alleles: list[str],
        n_flanks: list[str] | None = None,
        c_flanks: list[str] | None = None,
    ) -> pd.DataFrame:
        """Predict presentation scores with optional flanking sequences.

        Args:
            peptides: List of amino acid sequences.
            alleles: List of HLA allele strings.
            n_flanks: Optional N-terminal flanking sequences.
            c_flanks: Optional C-terminal flanking sequences.

        Returns:
            DataFrame with prediction results.
        """
        ...

    @property
    def name(self) -> str:
        """Predictor name.

        Returns:
            Name string.
        """
        ...
