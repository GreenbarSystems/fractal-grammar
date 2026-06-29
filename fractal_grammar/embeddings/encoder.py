"""
embeddings/encoder.py

Encodes BehavioralEvents into fixed-dimension float vectors.
Supports three backends, selected automatically by what's available:

  1. SentenceTransformer  — best quality, requires model download (~90MB)
  2. TF-IDF + SVD         — fast, zero download, lower quality
  3. Hash projection       — instant, deterministic, lowest quality

All backends produce L2-normalized vectors of a configurable dimension.
The downstream clustering and grammar layers are backend-agnostic.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from fractal_grammar.core.sequence import BehavioralEvent, BehavioralSequence

logger = logging.getLogger(__name__)


class Backend(Enum):
    SENTENCE_TRANSFORMER = auto()
    TFIDF_SVD            = auto()
    HASH_PROJECTION      = auto()
    HDC                  = auto()   # Hyperdimensional Computing — 10,000-dim bipolar


@dataclass
class EncodedEvent:
    event  : BehavioralEvent
    vector : NDArray[np.float32]   # shape (dim,), L2-normalized

    @property
    def fingerprint(self) -> str:
        return self.event.fingerprint


@dataclass
class EncodedSequence:
    sequence : BehavioralSequence
    events   : List[EncodedEvent] = field(default_factory=list)

    @property
    def matrix(self) -> NDArray[np.float32]:
        """Shape (n_events, dim). Convenient for clustering."""
        if not self.events:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([e.vector for e in self.events])

    def __len__(self) -> int:
        return len(self.events)


class BehavioralEncoder:
    """
    Encodes raw BehavioralEvents into continuous vector space.

    Parameters
    ----------
    dim      : target embedding dimension (must be <= model native dim)
    backend  : force a specific backend; None = auto-detect
    model    : SentenceTransformer model name (if using that backend)
    """

    def __init__(
        self,
        dim    : int = 128,
        backend: Optional[Backend] = None,
        model  : str = "all-MiniLM-L6-v2",
    ):
        self.dim     = dim
        self.model   = model
        self.backend = backend or self._detect_backend()
        self._encoder = None
        self._projection: Optional[NDArray[np.float32]] = None

        logger.info(f"BehavioralEncoder using backend: {self.backend.name}")
        self._initialize()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _detect_backend(self) -> Backend:
        try:
            import sentence_transformers  # noqa: F401
            return Backend.SENTENCE_TRANSFORMER
        except ImportError:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            return Backend.TFIDF_SVD
        except ImportError:
            pass
        return Backend.HDC  # HDC requires only numpy — always available

    def _initialize(self) -> None:
        if self.backend == Backend.SENTENCE_TRANSFORMER:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model)
            native_dim = self._encoder.get_sentence_embedding_dimension()
            if self.dim < native_dim:
                rng = np.random.default_rng(42)
                self._projection = rng.standard_normal(
                    (native_dim, self.dim)
                ).astype(np.float32)
                # Orthonormalize for stable projection
                self._projection, _ = np.linalg.qr(self._projection)
                self._projection = self._projection.T  # (dim, native_dim)
        elif self.backend == Backend.TFIDF_SVD:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.pipeline import Pipeline
            self._encoder = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=4096, sublinear_tf=True)),
                ("svd",   TruncatedSVD(n_components=self.dim, random_state=42)),
            ])
            self._fitted = False
        elif self.backend == Backend.HASH_PROJECTION:
            rng = np.random.default_rng(42)
            self._projection = rng.standard_normal(
                (4096, self.dim)
            ).astype(np.float32)
        elif self.backend == Backend.HDC:
            from fractal_grammar.embeddings.hdc_encoder import HDCEncoder
            self._hdc_encoder = HDCEncoder(dim=self.dim)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> NDArray[np.float32]:
        """Encode a single string to a normalized float32 vector."""
        if self.backend == Backend.SENTENCE_TRANSFORMER:
            vec = self._encoder.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            if self._projection is not None:
                vec = self._projection @ vec
        elif self.backend == Backend.TFIDF_SVD:
            if not self._fitted:
                raise RuntimeError(
                    "TF-IDF backend must be fit before encoding. "
                    "Call fit() with a corpus first."
                )
            vec = self._encoder.transform([text])[0].astype(np.float32)
        elif self.backend == Backend.HASH_PROJECTION:
            vec = self._hash_project(text)
        elif self.backend == Backend.HDC:
            _, vec = self._hdc_encoder.encode_text(text)

        return self._normalize(vec)

    def encode_batch(self, texts: List[str]) -> NDArray[np.float32]:
        """Encode multiple strings efficiently. Returns shape (n, dim)."""
        if self.backend == Backend.SENTENCE_TRANSFORMER:
            vecs = self._encoder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            ).astype(np.float32)
            if self._projection is not None:
                vecs = (self._projection @ vecs.T).T
        elif self.backend == Backend.TFIDF_SVD:
            if not self._fitted:
                raise RuntimeError("TF-IDF backend not fitted.")
            vecs = self._encoder.transform(texts).astype(np.float32)
        elif self.backend == Backend.HASH_PROJECTION:
            vecs = np.vstack([self._hash_project(t) for t in texts])
        elif self.backend == Backend.HDC:
            _, float_matrix = self._hdc_encoder.encode_batch(texts)
            return float_matrix  # already normalized

        return np.vstack([self._normalize(v) for v in vecs])

    def fit(self, texts: List[str]) -> "BehavioralEncoder":
        """Fit TF-IDF backend on a corpus. No-op for other backends."""
        if self.backend == Backend.TFIDF_SVD:
            self._encoder.fit(texts)
            self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Sequence encoding
    # ------------------------------------------------------------------

    def encode_sequence(self, seq: BehavioralSequence) -> EncodedSequence:
        """Encode every event in a sequence."""
        texts = [e.content for e in seq.events]
        if not texts:
            return EncodedSequence(sequence=seq)

        if self.backend == Backend.TFIDF_SVD and not self._fitted:
            self.fit(texts)

        vecs = self.encode_batch(texts)
        encoded = EncodedSequence(sequence=seq)
        for event, vec in zip(seq.events, vecs):
            encoded.events.append(EncodedEvent(event=event, vector=vec))
        return encoded

    def encode_sequences(
        self, sequences: List[BehavioralSequence]
    ) -> List[EncodedSequence]:
        """Fit (if needed) on all text and encode all sequences."""
        if self.backend == Backend.TFIDF_SVD and not self._fitted:
            all_texts = [e.content for seq in sequences for e in seq.events]
            self.fit(all_texts)
        return [self.encode_sequence(seq) for seq in sequences]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash_project(self, text: str) -> NDArray[np.float32]:
        """Deterministic random projection via SHA-256 hash features."""
        digest = hashlib.sha256(text.encode()).digest()
        # Repeat digest to fill 4096 bits
        expanded = (digest * 128)[:512]
        bits = np.unpackbits(np.frombuffer(expanded, dtype=np.uint8))[:4096]
        vec  = bits.astype(np.float32) * 2 - 1   # {-1, +1}
        return self._projection.T @ vec            # (dim,)

    @staticmethod
    def _normalize(vec: NDArray[np.float32]) -> NDArray[np.float32]:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec
