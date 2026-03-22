"""Agretopicity scoring for neoantigen candidates.

Agretopicity measures how much better a mutant peptide binds MHC compared
to its wildtype counterpart. A high agretopicity indicates the mutation
conferred a substantial binding advantage and the peptide is unlikely to
induce self-tolerance.
"""

from __future__ import annotations

import logging

import pandas as pd


class AgretopicityScorer:
    """Computes agretopicity scores for mutant/wildtype peptide pairs.

    Agretopicity is defined as the ratio of wildtype to mutant binding
    affinity (in nM IC50). Because lower IC50 = stronger binding, a ratio
    greater than 1.0 means the mutant binds more strongly than the wildtype.

    Reference:
        Duan et al. (2014) Cancer Immunology Research.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(type(self).__qualname__)

    def compute(self, mutant_affinity_nm: float, wildtype_affinity_nm: float) -> float:
        """Compute agretopicity for a single peptide pair.

        Args:
            mutant_affinity_nm: Mutant peptide MHC binding affinity (IC50, nM).
                Lower values indicate stronger binding.
            wildtype_affinity_nm: Wildtype peptide MHC binding affinity (IC50, nM).

        Returns:
            Agretopicity score: wildtype_affinity / mutant_affinity.
            Returns 0.0 if mutant_affinity is 0 to avoid division by zero.
            A score > 1.0 indicates the mutant binds more strongly.
        """
        if mutant_affinity_nm == 0.0:
            self._logger.debug("Mutant affinity is 0 nM; returning agretopicity=0.0")
            return 0.0
        return wildtype_affinity_nm / mutant_affinity_nm

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add an agretopicity column to a prediction DataFrame.

        The input DataFrame must contain:
        - ``mhcflurry_affinity``: mutant peptide IC50 in nM
        - ``wildtype_affinity``: wildtype peptide IC50 in nM

        Args:
            df: Prediction DataFrame with mutant and wildtype affinities.

        Returns:
            A copy of the DataFrame with an added ``agretopicity`` column.

        Raises:
            KeyError: If required columns are absent.
        """
        required = {"mhcflurry_affinity", "wildtype_affinity"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"AgretopicityScorer.annotate_dataframe requires columns "
                f"{required}; missing: {missing}"
            )

        df = df.copy()
        df["agretopicity"] = df.apply(
            lambda row: self.compute(
                row["mhcflurry_affinity"], row["wildtype_affinity"]
            ),
            axis=1,
        )
        return df
