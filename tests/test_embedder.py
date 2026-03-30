"""
Tests for bll/embedder.py — SentenceTransformerEmbedder

Test split:
  Unit tests (no model required):
    - test_model_load_error: verifies ModelLoadError on invalid model name.
      Runs in any environment — does not need cached model.

  Integration tests (@pytest.mark.integration — require cached embedding model):
    - test_batch_embedding_shape: verifies output shape (n, d).
    - test_embedding_l2_normalized: verifies L2-norm ≈ 1.0 per vector.
    - test_embed_batch_empty_list: verifies graceful handling of empty input.
    - test_get_model_name: verifies model identifier is returned correctly.

Run unit tests only (CI, no model):
    pytest tests/test_embedder.py -m "not integration"

Run all (model must be cached):
    pytest -m integration tests/test_embedder.py
"""

import numpy as np
import pytest

from bll.embedder import ModelLoadError, SentenceTransformerEmbedder

# Default model used throughout PoC2 (see config/versions/v1.0-default.yaml)
DEFAULT_MODEL = "Vsevolod/company-names-similarity-sentence-transformer"


# ---------------------------------------------------------------------------
# Unit tests — no model required
# ---------------------------------------------------------------------------

def test_model_load_error():
    """Non-existent model path must raise ModelLoadError with an actionable message."""
    with pytest.raises(ModelLoadError) as exc_info:
        SentenceTransformerEmbedder("this-model-does-not-exist/xyz-abc-999")

    # Error message must include model name for auditability
    assert "this-model-does-not-exist/xyz-abc-999" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration tests — require cached embedding model
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_batch_embedding_shape():
    """embed_batch() returns shape (n, d) where n == len(names) and d > 0."""
    names = ["Deutsche Bank AG", "Bayerische Landesbank", "ACME GmbH"]
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    result = embedder.embed_batch(names)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3, f"Expected 3 rows, got {result.shape[0]}"
    assert result.shape[1] > 0, f"Expected positive embedding dimension, got {result.shape[1]}"


@pytest.mark.integration
def test_embedding_l2_normalized():
    """All embedding vectors must be L2-normalized (norm ≈ 1.0, atol=1e-5)."""
    names = ["Deutsche Bank AG", "Allianz SE", "Siemens Aktiengesellschaft"]
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    result = embedder.embed_batch(names)

    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(
        norms,
        1.0,
        atol=1e-5,
        err_msg=f"L2 norms not close to 1.0: {norms}",
    )


@pytest.mark.integration
def test_embed_batch_empty_list():
    """embed_batch() with empty input returns shape (0, d) — no crash."""
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    result = embedder.embed_batch([])

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 0
    assert result.shape[1] > 0, "Embedding dimension must be positive even for empty input"


@pytest.mark.integration
def test_get_model_name():
    """get_model_name() returns the model identifier passed at construction."""
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    assert embedder.get_model_name() == DEFAULT_MODEL


@pytest.mark.integration
def test_embeddings_are_float32():
    """Embeddings must be float32 (FAISS IndexFlatIP requirement)."""
    names = ["Bayerische Landesbank GmbH"]
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    result = embedder.embed_batch(names)

    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


@pytest.mark.integration
def test_cosine_similarity_same_name():
    """Cosine similarity of identical name embeddings must be ≈ 1.0."""
    name = "Deutsche Bank AG"
    embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    embeddings = embedder.embed_batch([name, name])

    # L2-normalized → inner product = cosine similarity
    cosine = float(np.dot(embeddings[0], embeddings[1]))
    assert cosine > 0.9999, f"Expected cosine ≈ 1.0 for identical names, got {cosine:.6f}"
