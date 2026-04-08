"""Simple CSV-based cache for prediction results."""

from __future__ import annotations

import os

import pandas as pd


class PredictionCache:
    """Simple CSV-based cache for prediction results.

    Stores and retrieves DataFrames keyed by a cache name and a hash
    of the input parameters (alleles, number of candidates, predictor name).

    Args:
        cache_dir: Directory for cache files. Created if it doesn't exist.
    """

    def __init__(self, cache_dir: str = "results/cache") -> None:
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> pd.DataFrame | None:
        """Load cached results, or return None if not cached."""
        path = os.path.join(self._cache_dir, key)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Store results to cache."""
        path = os.path.join(self._cache_dir, key)
        df.to_csv(path, index=False)

    def make_key(
        self, predictor_name: str, n_candidates: int, alleles: list[str]
    ) -> str:
        """Generate a deterministic cache key from prediction parameters.

        The key is human-readable so users can find and delete cache files
        manually if needed.

        Args:
            predictor_name: Short identifier for the predictor (e.g. "mhcflurry").
            n_candidates: Number of input peptide candidates.
            alleles: HLA allele list used in the prediction run.

        Returns:
            Filename string, e.g. ``"mhcflurry_8432cands_4alleles.csv"``.
        """
        n_alleles = len(alleles)
        return f"{predictor_name}_{n_candidates}cands_{n_alleles}alleles.csv"
