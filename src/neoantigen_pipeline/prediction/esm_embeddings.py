"""ESM-2 protein language model embedding module.

Computes per-residue embeddings for proteins using ESM-2 and caches them in
an HDF5 file keyed by transcript ID. Subsequent pipeline runs load from cache
without recomputation.

ESM-2 and PyTorch are optional dependencies. They are imported lazily and a
clear error is raised if they are missing when features are requested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from neoantigen_pipeline.exceptions import PredictorNotInstalledError

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

# ESM-2 max context length (tokens, excluding BOS/EOS)
_ESM_MAX_LEN = 1022

try:
    import esm as _esm_lib

    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def _require_esm() -> None:
    if not ESM_AVAILABLE:
        raise PredictorNotInstalledError(
            "ESM-2 features require the 'esm' extra. "
            "Install with: pip install 'neoantigen-pipeline[esm]'"
        )


def _require_h5py() -> None:
    if not H5PY_AVAILABLE:
        raise PredictorNotInstalledError(
            "ESM embedding cache requires h5py. Install with: pip install h5py"
        )


class ESMEmbeddingCache:
    """Computes and caches ESM-2 per-residue embeddings for proteins.

    Embeddings are computed once per unique protein and stored on disk as an
    HDF5 file keyed by transcript ID (version suffix stripped). Subsequent
    runs load from cache without recomputation.

    Args:
        model_name: ESM-2 model identifier, e.g. ``"esm2_t33_650M_UR50D"``.
        cache_path: Path to the HDF5 cache file. Created on first write.
        device: PyTorch device string. Use ``"auto"`` to select CUDA if
            available, otherwise CPU.
        batch_size: Number of proteins to process per forward pass.
        context_window: Residues on each side of the peptide for context
            window features.
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        cache_path: str = "results/esm_cache.h5",
        device: str = "auto",
        batch_size: int = 4,
        context_window: int = 15,
    ) -> None:
        _require_esm()
        _require_h5py()

        self._model_name = model_name
        self._cache_path = cache_path
        self._context_window = context_window
        self._batch_size = batch_size
        self._device = self._resolve_device(device)
        self._model: object | None = None
        self._alphabet: object | None = None
        self._repr_layer: int | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def get_embedding(self, transcript_id: str, sequence: str) -> np.ndarray:
        """Return per-residue ESM-2 embeddings for a protein sequence.

        Loads from cache if available; otherwise computes and stores.

        Args:
            transcript_id: Ensembl transcript ID (version suffix stripped).
            sequence: Full amino acid sequence of the protein.

        Returns:
            NumPy array of shape ``(seq_len, embed_dim)`` where ``embed_dim``
            is 1280 for ``esm2_t33_650M_UR50D``.
        """
        key = transcript_id.split(".")[0]
        with h5py.File(self._cache_path, "a") as f:
            if key in f:
                return f[key][:]
            embedding = self._compute_embedding(sequence)
            f.create_dataset(key, data=embedding, compression="gzip")
            return embedding

    def precompute(self, proteins: dict[str, str]) -> None:
        """Precompute and cache embeddings for all proteins not already cached.

        Args:
            proteins: Mapping from transcript_id to amino acid sequence.
        """
        with h5py.File(self._cache_path, "a") as f:
            missing = {
                tid: seq for tid, seq in proteins.items() if tid.split(".")[0] not in f
            }

        if not missing:
            _logger.info("All %d proteins already cached.", len(proteins))
            return

        _logger.info(
            "Computing ESM-2 embeddings for %d/%d uncached proteins.",
            len(missing),
            len(proteins),
        )

        # Sort by length ascending so shorter sequences pad less within a batch
        sorted_items = sorted(missing.items(), key=lambda kv: len(kv[1]))
        batches = [
            sorted_items[i : i + self._batch_size]
            for i in range(0, len(sorted_items), self._batch_size)
        ]

        with h5py.File(self._cache_path, "a") as f:
            for batch in batches:
                for tid, seq in batch:
                    key = tid.split(".")[0]
                    if key in f:
                        continue
                    embedding = self._compute_embedding(seq)
                    f.create_dataset(key, data=embedding, compression="gzip")
                    _logger.debug("Cached embedding for %s (len=%d)", key, len(seq))

    def extract_peptide_features(
        self,
        transcript_id: str,
        sequence: str,
        start: int,
        end: int,
        mutation_pos: int | None = None,
    ) -> np.ndarray:
        """Extract pooled ESM-2 features for a peptide region.

        Computes three feature vectors and concatenates them:

        1. **Peptide mean**: mean-pooled embedding over ``[start:end]``.
        2. **Context mean**: mean-pooled embedding over a ±``context_window``
           residue window around the peptide.
        3. **Mutation residue**: embedding at ``mutation_pos`` (or centre of
           peptide if ``mutation_pos`` is ``None``).

        Args:
            transcript_id: Protein identifier for cache lookup.
            sequence: Full protein sequence.
            start: 0-based start index of the peptide in the protein.
            end: 0-based exclusive end index of the peptide.
            mutation_pos: 0-based position of the mutated residue in the
                protein. Defaults to the centre of ``[start:end]``.

        Returns:
            Feature vector of shape ``(embed_dim * 3,)`` — concatenation of
            peptide mean, context mean, and mutation residue embedding.
        """
        embedding = self.get_embedding(transcript_id, sequence)
        seq_len = embedding.shape[0]

        # Peptide region
        pep_start = max(0, start)
        pep_end = min(seq_len, end)
        peptide_mean = embedding[pep_start:pep_end].mean(axis=0)

        # Context window
        ctx_start = max(0, start - self._context_window)
        ctx_end = min(seq_len, end + self._context_window)
        context_mean = embedding[ctx_start:ctx_end].mean(axis=0)

        # Mutation residue
        if mutation_pos is None:
            mutation_pos = (start + end) // 2
        mut_idx = max(0, min(seq_len - 1, mutation_pos))
        mutation_residue = embedding[mut_idx]

        return np.concatenate([peptide_mean, context_mean, mutation_residue])

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' based on availability."""
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> tuple[object, object, int]:
        """Load and cache the ESM-2 model, alphabet, and representation layer.

        Returns:
            Tuple of ``(model, batch_converter, repr_layer)``.
        """
        if self._model is not None:
            return self._model, self._alphabet, self._repr_layer  # type: ignore[return-value]

        _logger.info(
            "Loading ESM-2 model '%s' on device '%s'.", self._model_name, self._device
        )
        model, alphabet = getattr(_esm_lib.pretrained, self._model_name)()
        model = model.eval().to(self._device)

        # Determine the last representation layer index
        repr_layer = model.num_layers

        self._model = model
        self._alphabet = alphabet
        self._repr_layer = repr_layer
        return model, alphabet, repr_layer

    def _compute_embedding(self, sequence: str) -> np.ndarray:
        """Compute per-residue ESM-2 embedding for a single protein sequence.

        Sequences longer than ``_ESM_MAX_LEN`` are truncated with a warning.

        Args:
            sequence: Full protein amino acid sequence.

        Returns:
            NumPy array of shape ``(seq_len, embed_dim)``.
        """
        import torch

        if len(sequence) > _ESM_MAX_LEN:
            _logger.warning(
                "Protein sequence of length %d exceeds ESM-2 max length %d; "
                "truncating to %d residues.",
                len(sequence),
                _ESM_MAX_LEN,
                _ESM_MAX_LEN,
            )
            sequence = sequence[:_ESM_MAX_LEN]

        model, alphabet, repr_layer = self._load_model()
        batch_converter = alphabet.get_batch_converter()  # type: ignore[attr-defined]

        data = [("protein", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(self._device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)  # type: ignore[operator]

        # Shape: [1, seq_len+2, embed_dim] — strip BOS and EOS tokens
        token_repr = results["representations"][repr_layer]
        embedding = token_repr[0, 1 : len(sequence) + 1].cpu().numpy()
        return embedding
