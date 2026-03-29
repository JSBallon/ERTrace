"""
BLL — SentenceTransformerEmbedder

Wraps sentence-transformers for batch embedding of company names.
Produces L2-normalized float32 vectors for cosine similarity via FAISS IndexFlatIP.

Design decisions (see ADR-M2-001):
  - Single model.encode() call for all names (batch pattern — no per-name loop)
  - normalize_embeddings=True: L2-norm happens inside the model call
    → enables exact cosine similarity via inner product in FAISS IndexFlatIP
  - batch_size=256: optimal for CPU inference at PoC scale
  - show_progress_bar=False: progress is reported at pipeline level via callback
  - ModelLoadError on missing/invalid model path: clean fail-fast with hint

No Streamlit imports. No DAL access. No filesystem access. No external API calls
after the initial one-time model download.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ModelLoadError(Exception):
    """
    Raised when the embedding model cannot be loaded.

    Provides a clean, actionable error message with the model name and
    a download hint — logged as a run_start failure event by the pipeline.
    """
    pass


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """
    Batch encoder for company names using a locally cached Sentence Transformer model.

    Usage:
        embedder = SentenceTransformerEmbedder("Vsevolod/company-names-similarity-sentence-transformer")
        embeddings = embedder.embed_batch(["Deutsche Bank AG", "Allianz SE"])
        # embeddings.shape == (2, d), L2-normalized

    The model is loaded once at construction. All subsequent encode calls are local
    — no network access after the initial one-time HuggingFace download.
    """

    def __init__(self, model_name: str) -> None:
        """
        Load the Sentence Transformer model.

        Args:
            model_name: HuggingFace model identifier or local path.
                        E.g. "Vsevolod/company-names-similarity-sentence-transformer"

        Raises:
            ModelLoadError: If the model cannot be loaded (not found locally or
                            on HuggingFace, or the path is invalid).
        """
        self._model_name = model_name
        try:
            self._model = SentenceTransformer(model_name)
        except (OSError, ValueError, Exception) as exc:
            raise ModelLoadError(
                f"Cannot load embedding model '{model_name}'. "
                f"If the model is not yet cached locally, run: "
                f"python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{model_name}')\" "
                f"to trigger the one-time download (~500 MB for the default model). "
                f"Original error: {exc}"
            ) from exc

    def embed_batch(self, names: list[str]) -> np.ndarray:
        """
        Batch-encode all names in a single model call.

        Applies L2-normalization inside the model call (normalize_embeddings=True),
        which is the prerequisite for cosine similarity via FAISS IndexFlatIP.

        Args:
            names: List of normalized company name strings.
                   Normalization (legal form stripping, unicode, lowercase) must
                   be applied by the DAL before calling this method.

        Returns:
            np.ndarray of shape (n, d), dtype float32, L2-normalized.
            Each row is the embedding vector for names[i].

        Notes:
            - Empty list returns shape (0, d) where d is the model dimension.
            - show_progress_bar=False: pipeline-level progress callback is used instead.
            - batch_size=256: optimal for CPU; reduces memory pressure vs. larger batches.
        """
        if not names:
            # Return empty array with correct embedding dimension
            dim = self._model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)

        embeddings = self._model.encode(
            names,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-norm in model → cosine via inner product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def get_model_name(self) -> str:
        """
        Return the active model identifier for audit logging.

        Captured in the run_start event (embedding_model field) so every
        run is traceable to the exact model version used.
        """
        return self._model_name
