"""
BLL — FaissSearcher

FAISS-based cosine similarity search for the TGFR pipeline (Stage 1 output).

Design decisions (see ADR-M2-002):
  - IndexFlatIP: exact inner product = exact cosine for L2-normalized vectors
    → deterministic, reproducible, no training step, no approximation error
  - Single batch index.search() call — no per-row loop (batch efficiency pattern)
  - Score clipping to [0.0, 1.0] in get_candidate(): float32 precision can produce
    values like 1.0000001 or -0.0000003 even for correctly L2-normalized vectors
  - FAISS sentinel (-1) filter: handles top_k > len(B) without crashing
  - Index rebuilt per run (in-memory, no persistence) — correct for PoC;
    production path uses Qdrant with persistent index

Prerequisite: embeddings_b passed to __init__ MUST be L2-normalized float32.
The caller (pipeline.py) is responsible for ensuring this via SentenceTransformerEmbedder.

No Streamlit imports. No DAL access. No filesystem access. No external API calls.
"""

import numpy as np
import faiss


class FaissSearcher:
    """
    FAISS IndexFlatIP similarity searcher for Top-K candidate retrieval.

    Builds the index at construction from Source B embeddings.
    All search and candidate retrieval operations are stateless after construction.

    Usage:
        searcher = FaissSearcher(embeddings_b)                  # build index
        scores, indices = searcher.search(embeddings_a, top_k=5) # batch search
        candidates = searcher.get_candidate(scores, indices, i=0) # per-entry list
    """

    def __init__(self, embeddings_b: np.ndarray) -> None:
        """
        Build a FAISS IndexFlatIP from Source B embeddings.

        The index is built immediately at construction — no separate build step.
        All vectors are copied into the FAISS index; the original array can be
        discarded after construction if memory is a concern.

        Args:
            embeddings_b: L2-normalized float32 array of shape (m, d).
                          Must satisfy: np.linalg.norm(embeddings_b, axis=1) ≈ 1.0
                          Produced by SentenceTransformerEmbedder.embed_batch().

        Raises:
            ValueError: If embeddings_b is empty (0 rows) or not 2-dimensional.
        """
        if embeddings_b.ndim != 2:
            raise ValueError(
                f"embeddings_b must be 2-dimensional (m, d), got shape {embeddings_b.shape}"
            )
        if embeddings_b.shape[0] == 0:
            raise ValueError(
                "embeddings_b must contain at least one vector (shape[0] > 0). "
                "Source B cannot be empty."
            )

        # Ensure float32 — FAISS requires float32
        embeddings_b = embeddings_b.astype(np.float32)

        d = embeddings_b.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(embeddings_b)   # adds all m vectors in one call
        self._n_total = embeddings_b.shape[0]

    def search(
        self,
        embeddings_a: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch-search all Source A entries against the Source B index.

        Performs a single index.search() call for all A entries — no per-row loop.
        Returns raw FAISS output (not yet clipped or filtered) so that the caller
        can inspect or store the full matrix before extracting per-entry candidates.

        Args:
            embeddings_a: L2-normalized float32 array of shape (n, d).
            top_k:        Number of nearest neighbors to retrieve per A entry.
                          Clamped internally to min(top_k, n_total) to avoid FAISS errors.

        Returns:
            scores:  np.ndarray shape (n, top_k), raw cosine scores (float32).
                     May contain values outside [0.0, 1.0] due to float32 precision.
                     Values of -1.0 at position j mean fewer than j+1 results exist.
            indices: np.ndarray shape (n, top_k), int64 indices into Source B.
                     Value -1 is FAISS sentinel for "no result at this position".
        """
        # Clamp top_k to the number of indexed vectors — FAISS errors if k > ntotal
        effective_k = min(top_k, self._n_total)

        # Ensure float32
        embeddings_a = embeddings_a.astype(np.float32)

        # ✅ Single batch call — mandatory (no per-row loop)
        scores, indices = self._index.search(embeddings_a, k=effective_k)
        # scores.shape == indices.shape == (n, effective_k)
        return scores, indices

    def get_candidate(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        i: int,
    ) -> list[tuple[int, float]]:
        """
        Extract the Top-K candidates for Source A entry i as a clean list.

        Clips cosine scores to [0.0, 1.0] (float32 precision defense).
        Filters out FAISS sentinel indices (-1) that appear when top_k > len(B).
        Returns results in descending score order (FAISS already guarantees this
        for IndexFlatIP — no re-sort needed).

        Args:
            scores:  Raw score matrix from search(), shape (n, top_k).
            indices: Index matrix from search(), shape (n, top_k).
            i:       Row index for Source A entry to extract.

        Returns:
            List of (b_index, cosine_score) tuples, sorted descending by cosine_score.
            b_index is the position of the candidate in Source B.
            cosine_score is clipped to [0.0, 1.0].
            Empty list if no valid candidates exist for entry i.
        """
        candidates = []
        for j in range(scores.shape[1]):
            b_idx = int(indices[i, j])
            if b_idx == -1:
                # FAISS sentinel: top_k exceeded the number of indexed vectors
                continue
            cosine_score = float(np.clip(scores[i, j], 0.0, 1.0))
            candidates.append((b_idx, cosine_score))
        return candidates

    @property
    def n_total(self) -> int:
        """Number of Source B vectors in the index."""
        return self._n_total
