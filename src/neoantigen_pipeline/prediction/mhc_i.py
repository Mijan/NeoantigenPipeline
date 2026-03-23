"""Abstract MHC class I binding predictor interface.

Defines the MHC-I–specific prediction contract (processing-aware prediction
and wildtype prediction for agretopicity). Concrete backends such as
``MHCflurryPredictor`` subclass this.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from neoantigen_pipeline.prediction.base import BindingPredictor

if TYPE_CHECKING:
    from neoantigen_pipeline.candidates.peptide_generator import PeptideCandidate
    from neoantigen_pipeline.prediction.results import (
        MHCIPredictionResult,
        WildtypePredictionResult,
    )


class MHCIPredictor(BindingPredictor):
    """Abstract base class for MHC class I binding predictors.

    Extends ``BindingPredictor`` with two MHC-I–specific prediction methods:

    - ``predict_with_processing``: uses N/C-terminal flanking sequences from
      ``PeptideCandidate`` objects to improve antigen-processing prediction.
    - ``predict_wildtype``: predicts binding for the wildtype counterpart of
      each mutant candidate, used to compute agretopicity.

    Concrete implementations (e.g. ``MHCflurryPredictor``) override all three
    abstract methods from this class and ``BindingPredictor``.
    """

    @property
    def mhc_class(self) -> int:
        """MHC class.

        Returns:
            1 (class I)
        """
        return 1

    @abstractmethod
    def predict_with_processing(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> list[MHCIPredictionResult]:
        """Predict MHC-I presentation with antigen-processing context.

        Passes N- and C-terminal flanking sequences from each candidate to the
        underlying model for improved processing prediction accuracy.

        Args:
            candidates: ``PeptideCandidate`` objects with sequence and flank data.
            alleles: HLA-I allele strings to predict against.

        Returns:
            List of ``MHCIPredictionResult`` objects, one per input candidate,
            in the same order as ``candidates``.

        Raises:
            PredictionError: If prediction fails.
        """

    @abstractmethod
    def predict_wildtype(
        self,
        candidates: list[PeptideCandidate],
        alleles: list[str],
    ) -> list[WildtypePredictionResult]:
        """Predict MHC-I binding for the wildtype counterpart of each candidate.

        Used to compute agretopicity (wildtype IC50 / mutant IC50). The result
        at index *i* is paired with ``predict_with_processing`` result at
        index *i* — both derive from ``candidates[i]``.

        Args:
            candidates: ``PeptideCandidate`` objects whose ``wildtype_sequence``
                fields are used as query peptides.
            alleles: HLA-I allele strings.

        Returns:
            List of ``WildtypePredictionResult`` objects, one per input
            candidate, in the same order as ``candidates``.

        Raises:
            PredictionError: If prediction fails.
        """
