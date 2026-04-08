"""Tests for the ESM-2 protein embedding module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neoantigen_pipeline.exceptions import PredictorNotInstalledError

# ── Helpers ──────────────────────────────────────────────────────────────────

_EMBED_DIM = 1280
_SEQ = "ACDEFGHIKLMNPQRSTVWY" * 5  # 100 aa test sequence


def _fake_embedding(seq: str) -> np.ndarray:
    """Return a deterministic random embedding for a sequence."""
    rng = np.random.default_rng(len(seq))
    return rng.random((len(seq), _EMBED_DIM)).astype(np.float32)


# ── Guard: skip all tests if ESM or h5py not installed ───────────────────────

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _esm_available() -> bool:
    try:
        import esm  # noqa: F401
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False


skip_without_esm = pytest.mark.skipif(
    not _esm_available(),
    reason="fair-esm and h5py not installed",
)


# ── Import guard ─────────────────────────────────────────────────────────────


class TestImportGuard:
    def test_raises_without_esm(self, tmp_path):
        with patch.dict("sys.modules", {"esm": None}):
            # Force re-evaluation of ESM_AVAILABLE

            import neoantigen_pipeline.prediction.esm_embeddings as mod

            original = mod.ESM_AVAILABLE
            mod.ESM_AVAILABLE = False
            try:
                from neoantigen_pipeline.prediction.esm_embeddings import (
                    ESMEmbeddingCache,
                )

                with pytest.raises(PredictorNotInstalledError, match="esm"):
                    ESMEmbeddingCache(cache_path=str(tmp_path / "x.h5"))
            finally:
                mod.ESM_AVAILABLE = original

    def test_raises_without_h5py(self, tmp_path):
        import neoantigen_pipeline.prediction.esm_embeddings as mod

        original_esm = mod.ESM_AVAILABLE
        original_h5py = mod.H5PY_AVAILABLE
        mod.ESM_AVAILABLE = True
        mod.H5PY_AVAILABLE = False
        try:
            from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

            with pytest.raises(PredictorNotInstalledError, match="h5py"):
                ESMEmbeddingCache(cache_path=str(tmp_path / "x.h5"))
        finally:
            mod.ESM_AVAILABLE = original_esm
            mod.H5PY_AVAILABLE = original_h5py


# ── Unit tests with mocked ESM ────────────────────────────────────────────────


@skip_without_esm
class TestCacheReadWrite:
    def test_embedding_written_on_first_call(self, tmp_path):
        import h5py

        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        cache._compute_embedding = MagicMock(return_value=_fake_embedding(_SEQ))

        cache.get_embedding("ENST001", _SEQ)

        with h5py.File(str(tmp_path / "cache.h5"), "r") as f:
            assert "ENST001" in f

    def test_embedding_loaded_from_cache(self, tmp_path):
        import h5py

        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        expected = _fake_embedding(_SEQ)

        with h5py.File(str(tmp_path / "cache.h5"), "a") as f:
            f.create_dataset("ENST001", data=expected)

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        cache._compute_embedding = MagicMock()

        result = cache.get_embedding("ENST001.3", _SEQ)

        cache._compute_embedding.assert_not_called()
        np.testing.assert_array_almost_equal(result, expected)

    def test_version_suffix_stripped_from_key(self, tmp_path):
        import h5py

        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        cache._compute_embedding = MagicMock(return_value=_fake_embedding(_SEQ))

        cache.get_embedding("ENST001.7", _SEQ)

        with h5py.File(str(tmp_path / "cache.h5"), "r") as f:
            assert "ENST001" in f
            assert "ENST001.7" not in f


@skip_without_esm
class TestPrecompute:
    def test_only_missing_proteins_computed(self, tmp_path):
        import h5py

        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        # Pre-cache one protein
        existing = _fake_embedding("ACDE")
        with h5py.File(str(tmp_path / "cache.h5"), "a") as f:
            f.create_dataset("ENST001", data=existing)

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        cache._compute_embedding = MagicMock(return_value=_fake_embedding("GHIK"))

        cache.precompute({"ENST001": "ACDE", "ENST002": "GHIK"})

        # Only ENST002 should trigger computation
        cache._compute_embedding.assert_called_once()

    def test_all_cached_does_not_compute(self, tmp_path):
        import h5py

        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        with h5py.File(str(tmp_path / "cache.h5"), "a") as f:
            f.create_dataset("ENST001", data=_fake_embedding("ACDE"))

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        cache._compute_embedding = MagicMock()

        cache.precompute({"ENST001": "ACDE"})
        cache._compute_embedding.assert_not_called()


@skip_without_esm
class TestExtractPeptideFeatures:
    def test_output_shape(self, tmp_path):
        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        embedding = _fake_embedding(_SEQ)
        cache.get_embedding = MagicMock(return_value=embedding)

        features = cache.extract_peptide_features(
            transcript_id="ENST001",
            sequence=_SEQ,
            start=10,
            end=19,
        )
        assert features.shape == (_EMBED_DIM * 3,)

    def test_mutation_pos_defaults_to_centre(self, tmp_path):
        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        embedding = _fake_embedding(_SEQ)
        cache.get_embedding = MagicMock(return_value=embedding)

        # Should not raise
        features = cache.extract_peptide_features(
            transcript_id="ENST001",
            sequence=_SEQ,
            start=10,
            end=19,
            mutation_pos=None,
        )
        assert features.shape == (_EMBED_DIM * 3,)

    def test_explicit_mutation_pos(self, tmp_path):
        from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

        cache = ESMEmbeddingCache(cache_path=str(tmp_path / "cache.h5"), device="cpu")
        embedding = _fake_embedding(_SEQ)
        cache.get_embedding = MagicMock(return_value=embedding)

        features_default = cache.extract_peptide_features(
            "ENST001", _SEQ, start=10, end=19, mutation_pos=14
        )
        features_explicit = cache.extract_peptide_features(
            "ENST001", _SEQ, start=10, end=19, mutation_pos=14
        )
        np.testing.assert_array_equal(features_default, features_explicit)


# ── Integration test (skipped unless ESM models are downloaded) ───────────────


@pytest.mark.integration
@pytest.mark.gpu
@skip_without_esm
def test_esm_integration_real_model(tmp_path):
    """Run real ESM-2 inference on a short protein sequence."""
    from neoantigen_pipeline.prediction.esm_embeddings import ESMEmbeddingCache

    cache = ESMEmbeddingCache(
        model_name="esm2_t6_8M_UR50D",  # smallest model for CI speed
        cache_path=str(tmp_path / "cache.h5"),
        device="cpu",
    )
    seq = "ACDEFGHIKLMNPQRSTVWY"
    embedding = cache.get_embedding("ENST_TEST", seq)

    assert embedding.ndim == 2
    assert embedding.shape[0] == len(seq)
