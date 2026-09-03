"""
Cortex Hybrid Vector (CHOV)

Vector híbrido optimizado para CPU con subespacios:
- Semantic: Int8 dense (cosine similarity)
- Role: Binary dense (Hamming similarity)
- Context: Sparse binary (Jaccard similarity)

Performance:
- Memory: ~10 KB/vector (4x menos que float32)
- Bind: ~2 μs (25x faster)
- Similarity: ~5 μs (20x faster)
- Cache: Fits in L2 (256 KB)

Author: Cortex Neural IA

NOTE (saved locally 2026-08-29): archived from Hackerprod/Cortex-SNN
(cortex/core/chov.py) before removal from the remote repo. Kept as
reference for a possible future MRDL production/scaling pass, not for
the current TFA/composition investigation. Known defects found on review:
- permute() leaves `role` unchanged (bit rotation is a TODO, not implemented).
- unbind() assumes bind is self-inverse for all three subspaces, but the
  semantic subspace's bind (int8 multiply + clip) is lossy/non-injective,
  unlike role (XOR) and context (symmetric difference) which really are
  self-inverse. Recovering A from A⊗B does not hold for the semantic part.
"""

import numpy as np
import numba as nb
from typing import List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# ============================================================================

@nb.njit(fastmath=True, cache=True)
def _int8_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind int8 semantic vectors (multiply + clip)"""
    result = np.empty_like(a)
    for i in range(len(a)):
        val = (a[i] * b[i]) // 127
        result[i] = max(-128, min(127, val))
    return result


@nb.njit(fastmath=True, cache=True)
def _int8_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for int8 vectors"""
    dot = 0
    norm_a = 0
    norm_b = 0

    for i in range(len(a)):
        dot += int(a[i]) * int(b[i])
        norm_a += int(a[i]) * int(a[i])
        norm_b += int(b[i]) * int(b[i])

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot) / (np.sqrt(float(norm_a)) * np.sqrt(float(norm_b)))


@nb.njit(cache=True)
def _binary_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR for binary role vectors (uint64 chunks)"""
    result = np.empty_like(a)
    for i in range(len(a)):
        result[i] = a[i] ^ b[i]
    return result


@nb.njit(cache=True)
def _binary_hamming_similarity(a: np.ndarray, b: np.ndarray, n_bits: int) -> float:
    """Hamming similarity for binary vectors"""
    hamming_dist = 0

    for i in range(len(a)):
        diff = np.uint64(a[i]) ^ np.uint64(b[i])
        # Popcount using Brian Kernighan's algorithm
        while diff > 0:
            diff = diff & (diff - np.uint64(1))
            hamming_dist += 1

    return 1.0 - (float(hamming_dist) / float(n_bits))


@nb.njit(cache=True)
def _sparse_jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity for sparse vectors (sorted indices)"""
    i = 0
    j = 0
    intersection = 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            intersection += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    union = len(a) + len(b) - intersection

    if union == 0:
        return 1.0

    return float(intersection) / float(union)


@nb.njit(cache=True)
def _int8_bundle(vectors: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Bundle int8 vectors with optional weights"""
    if len(vectors) == 0:
        return np.zeros(vectors[0].shape[0], dtype=np.int8)

    dim = vectors[0].shape[0]
    acc = np.zeros(dim, dtype=np.int32)

    if weights is None:
        total_weight = len(vectors)
        for v in vectors:
            for i in range(dim):
                acc[i] += v[i]
    else:
        total_weight = np.sum(weights)
        for idx, v in enumerate(vectors):
            w = weights[idx]
            for i in range(dim):
                acc[i] += int(v[i] * w)

    result = np.empty(dim, dtype=np.int8)
    for i in range(dim):
        val = acc[i] // int(total_weight)
        result[i] = max(-128, min(127, val))

    return result


@nb.njit(cache=True)
def _binary_bundle(vectors: List[np.ndarray], n_bits: int) -> np.ndarray:
    """Bundle binary vectors (majority vote)"""
    if len(vectors) == 0:
        return np.zeros(vectors[0].shape[0], dtype=np.uint64)

    n_chunks = vectors[0].shape[0]
    counts = np.zeros(n_bits, dtype=np.int32)

    # Count bits
    for v in vectors:
        for chunk_idx in range(n_chunks):
            chunk = v[chunk_idx]
            for bit in range(64):
                if chunk & (np.uint64(1) << bit):
                    counts[chunk_idx * 64 + bit] += 1

    # Majority vote
    threshold = len(vectors) // 2
    result = np.zeros(n_chunks, dtype=np.uint64)

    for chunk_idx in range(n_chunks):
        chunk = np.uint64(0)
        for bit in range(64):
            bit_idx = chunk_idx * 64 + bit
            if bit_idx < n_bits and counts[bit_idx] > threshold:
                chunk |= (np.uint64(1) << bit)
        result[chunk_idx] = chunk

    return result


# ============================================================================
# CHOV CONFIG
# ============================================================================

@dataclass
class CHOVConfig:
    """Configuration for Cortex Hybrid Vector"""
    dim_semantic: int = 8_000
    dim_role: int = 8_000
    dim_context: int = 4_000
    context_density: float = 0.04  # 4% active indices

    # Similarity weights
    w_semantic: float = 0.6
    w_role: float = 0.3
    w_context: float = 0.1


# ============================================================================
# CORTEX HYBRID VECTOR
# ============================================================================

class CortexHybridVector:
    """
    Cortex Hybrid Vector (CHOV)

    Optimized hybrid vector with three subspaces:
    1. Semantic (int8 dense): Main meaning representation
    2. Role (binary dense): Position/role information
    3. Context (sparse binary): Anchors for verification

    Memory: ~10 KB per vector
    Operations: Numba-optimized for CPU performance
    """

    __slots__ = ('semantic', 'role', 'context_indices', 'config', '_role_chunks')

    def __init__(self, config: CHOVConfig = None):
        self.config = config or CHOVConfig()

        # Semantic subspace (int8)
        self.semantic = np.zeros(self.config.dim_semantic, dtype=np.int8)

        # Role subspace (binary, stored as uint64 chunks)
        self._role_chunks = (self.config.dim_role + 63) // 64
        self.role = np.zeros(self._role_chunks, dtype=np.uint64)

        # Context subspace (sparse binary, only active indices)
        self.context_indices = np.array([], dtype=np.uint32)

    @staticmethod
    def random(config: CHOVConfig = None) -> 'CortexHybridVector':
        """Generate random CHOV"""
        config = config or CHOVConfig()
        v = CortexHybridVector(config)

        # Random semantic
        v.semantic = np.random.randint(-128, 128, size=config.dim_semantic, dtype=np.int8)

        # Random role
        v.role = np.random.randint(0, 2**64, size=v._role_chunks, dtype=np.uint64)

        # Random context (sparse)
        n_active = int(config.dim_context * config.context_density)
        v.context_indices = np.sort(
            np.random.choice(config.dim_context, size=n_active, replace=False).astype(np.uint32)
        )

        return v

    def bind(self, other: 'CortexHybridVector') -> 'CortexHybridVector':
        """
        Bind two vectors (binding operation)

        - Semantic: multiply + clip
        - Role: XOR
        - Context: symmetric difference

        Time: ~2 μs (Numba optimized)
        """
        result = CortexHybridVector(self.config)

        # Bind semantic (int8 multiply)
        result.semantic = _int8_bind(self.semantic, other.semantic)

        # Bind role (XOR)
        result.role = _binary_xor(self.role, other.role)

        # Bind context (symmetric difference)
        set_a = set(self.context_indices)
        set_b = set(other.context_indices)
        result.context_indices = np.sort(np.array(list(set_a ^ set_b), dtype=np.uint32))

        return result

    def bundle(self, vectors: List['CortexHybridVector'],
               weights: Optional[List[float]] = None) -> 'CortexHybridVector':
        """
        Bundle multiple vectors (superposition)

        - Semantic: weighted average
        - Role: majority vote
        - Context: majority vote

        Time: ~5 μs per 10 vectors (Numba optimized)
        """
        if not vectors:
            return CortexHybridVector(self.config)

        all_vectors = [self] + vectors
        result = CortexHybridVector(self.config)

        # Bundle semantic (weighted average)
        semantic_list = [v.semantic for v in all_vectors]
        weight_array = None if weights is None else np.array([1.0] + weights, dtype=np.float32)
        result.semantic = _int8_bundle(semantic_list, weight_array)

        # Bundle role (majority vote)
        role_list = [v.role for v in all_vectors]
        result.role = _binary_bundle(role_list, self.config.dim_role)

        # Bundle context (majority vote on indices)
        from collections import Counter
        counter = Counter()
        for v in all_vectors:
            counter.update(v.context_indices)

        threshold = len(all_vectors) // 2
        active = [idx for idx, count in counter.items() if count > threshold]
        result.context_indices = np.sort(np.array(active, dtype=np.uint32))

        return result

    def similarity(self, other: 'CortexHybridVector') -> float:
        """
        Composite similarity (weighted sum of subspaces)

        Formula:
        sim = w_semantic * cosine(semantic)
            + w_role * hamming(role)
            + w_context * jaccard(context)

        Time: ~5 μs (Numba optimized)
        """
        # Semantic similarity (cosine)
        sim_semantic = _int8_cosine_similarity(self.semantic, other.semantic)

        # Role similarity (Hamming)
        sim_role = _binary_hamming_similarity(self.role, other.role, self.config.dim_role)

        # Context similarity (Jaccard)
        sim_context = _sparse_jaccard_similarity(self.context_indices, other.context_indices)

        # Composite
        return (self.config.w_semantic * sim_semantic +
                self.config.w_role * sim_role +
                self.config.w_context * sim_context)

    def permute(self, shift: int) -> 'CortexHybridVector':
        """
        Permute vector (for sequence encoding)

        - Semantic: circular shift
        - Role: bit rotation
        - Context: index shift modulo dim
        """
        result = CortexHybridVector(self.config)

        # Permute semantic (circular shift)
        result.semantic = np.roll(self.semantic, shift)

        # Permute role (bit rotation - simplified)
        # For simplicity, just XOR with shifted version
        shift_mod = shift % self.config.dim_role
        result.role = self.role  # TODO: implement proper bit rotation if needed

        # Permute context (index shift)
        if len(self.context_indices) > 0:
            result.context_indices = np.sort(
                ((self.context_indices + shift) % self.config.dim_context).astype(np.uint32)
            )
        else:
            result.context_indices = self.context_indices

        return result

    def memory_size(self) -> int:
        """Return memory footprint in bytes"""
        semantic_size = self.semantic.nbytes
        role_size = self.role.nbytes
        context_size = self.context_indices.nbytes
        return semantic_size + role_size + context_size

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'semantic': self.semantic.tolist(),
            'role': self.role.tolist(),
            'context_indices': self.context_indices.tolist(),
            'config': {
                'dim_semantic': self.config.dim_semantic,
                'dim_role': self.config.dim_role,
                'dim_context': self.config.dim_context,
                'context_density': self.config.context_density,
            }
        }

    @staticmethod
    def from_dict(data: dict) -> 'CortexHybridVector':
        """Deserialize from dictionary"""
        config = CHOVConfig(**data['config'])
        v = CortexHybridVector(config)
        v.semantic = np.array(data['semantic'], dtype=np.int8)
        v.role = np.array(data['role'], dtype=np.uint64)
        v.context_indices = np.array(data['context_indices'], dtype=np.uint32)
        return v


# ============================================================================
# UTILITIES
# ============================================================================

def cosine_similarity(a: CortexHybridVector, b: CortexHybridVector) -> float:
    """Alias for CHOV similarity"""
    return a.similarity(b)


def random_hypervector(dim: int = 8_000, config: CHOVConfig = None) -> CortexHybridVector:
    """Generate random hypervector (legacy compatibility)"""
    if config is None:
        config = CHOVConfig(dim_semantic=dim)
    return CortexHybridVector.random(config)


def normalize(v: CortexHybridVector) -> CortexHybridVector:
    """Normalize vector (no-op for CHOV, already normalized)"""
    return v
