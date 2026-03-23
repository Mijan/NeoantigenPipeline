"""Agretopicity scoring for neoantigen candidates.

Agretopicity measures how much better a mutant peptide binds MHC compared
to its wildtype counterpart. A high agretopicity indicates the mutation
conferred a substantial binding advantage and the peptide is unlikely to
induce central tolerance.

Reference:
    Duan et al. (2014) Cancer Immunology Research.
"""

from __future__ import annotations

import logging

import numpy as np


class AgretopicityScorer:
    """Computes agretopicity scores for mutant/wildtype peptide pairs.

    Agretopicity is defined as the ratio of wildtype to mutant binding
    affinity (nM IC50). Because lower IC50 = stronger binding, a ratio
    greater than 1.0 means the mutant binds more strongly than the wildtype.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(type(self).__qualname__)

    def compute(self, mutant_affinity_nm: float, wildtype_affinity_nm: float) -> float:
        """Compute agretopicity for a single peptide pair.

        Args:
            mutant_affinity_nm: Mutant peptide IC50 in nM (lower = stronger binding).
            wildtype_affinity_nm: Wildtype peptide IC50 in nM.

        Returns:
            Agretopicity = wildtype_affinity / mutant_affinity.
            Returns 0.0 if mutant_affinity is 0 to avoid division by zero.
            A score > 1.0 indicates the mutant binds more strongly.
        """
        if mutant_affinity_nm == 0.0:
            self._logger.debug("Mutant affinity is 0 nM; returning agretopicity=0.0")
            return 0.0
        return wildtype_affinity_nm / mutant_affinity_nm

    def compute_batch(
        self,
        mutant_affinities_nm: list[float],
        wildtype_affinities_nm: list[float],
    ) -> list[float]:
        """Compute agretopicity for a batch of peptide pairs using vectorised arithmetic.

        Args:
            mutant_affinities_nm: List of mutant peptide IC50 values in nM.
            wildtype_affinities_nm: List of wildtype peptide IC50 values in nM.
                Must be the same length as ``mutant_affinities_nm``.

        Returns:
            List of agretopicity scores in the same order as the inputs.

        Raises:
            ValueError: If the input lists have different lengths.
        """
        if len(mutant_affinities_nm) != len(wildtype_affinities_nm):
            raise ValueError(
                f"mutant_affinities_nm (len {len(mutant_affinities_nm)}) and "
                f"wildtype_affinities_nm (len {len(wildtype_affinities_nm)}) "
                "must have the same length"
            )
        mut = np.asarray(mutant_affinities_nm, dtype=float)
        wt = np.asarray(wildtype_affinities_nm, dtype=float)
        # Where mutant affinity is zero, agretopicity is 0; otherwise wt / mut.
        scores = np.where(mut == 0.0, 0.0, wt / np.where(mut == 0.0, 1.0, mut))
        return scores.tolist()
