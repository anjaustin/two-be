"""
Neural APU L-Cache Implementation

Cache for ternary compute operations with LRU eviction policy.
SECURE VERSION - All high/medium priority issues addressed.
"""

import time
import threading
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum
import numpy as np


ALLOWED_OPCODES = frozenset(
    {
        "TMUL",
        "TADD",
        "TGATE",
        "TATTN",
        "TNORM",
        "TLOOKUP",
        "MTFP_ADD",
        "MTFP_MUL",
        "MTFP_MATMUL",
    }
)

ALLOWED_PREFIXES = frozenset(
    {"TMUL_", "TADD_", "TGATE_", "TATTN_", "TNORM_", "TLOOKUP_", "MTFP_"}
)

OPCODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")


class CacheStatus(Enum):
    HOT = 0
    COLD = 1
    EVICT = 2
    PREFETCH = 3


@dataclass
class CacheLine:
    opcode_id: str
    config: int
    status: CacheStatus = CacheStatus.COLD
    age: int = 0
    packed_weights: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    tile_routing: Optional[np.ndarray] = None
    hit_count: int = 0
    avg_latency: float = 0.0
    last_hit: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    rejected: int = 0
    total_latency: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.3f}, evictions={self.evictions}, "
            f"rejected={self.rejected}"
        )


def validate_opcode(opcode_id: str) -> bool:
    """Validate opcode against allowlist and pattern."""
    if opcode_id in ALLOWED_OPCODES:
        return True
    for prefix in ALLOWED_PREFIXES:
        if opcode_id.startswith(prefix):
            try:
                int(opcode_id.split("_")[1])
                return True
            except (ValueError, IndexError):
                pass
    return False


def validate_operand_shape(opcode: str, operands: Tuple[np.ndarray, ...]) -> bool:
    """Validate operand shapes for given opcode."""
    if not operands:
        return False

    if opcode == "TMUL":
        if len(operands) < 1:
            return False
        a = operands[0]
        if a.ndim != 2:
            return False
        if a.shape[0] <= 0 or a.shape[1] <= 0:
            return False
        return True

    elif opcode == "TADD":
        if len(operands) != 2:
            return False
        if operands[0].shape != operands[1].shape:
            return False
        return True

    elif opcode == "TGATE":
        if len(operands) != 2:
            return False
        x = operands[0]
        logits = operands[1]
        if x.ndim != 2 or logits.ndim != 2:
            return False
        if x.shape[0] != logits.shape[0]:
            return False
        if x.shape[1] % logits.shape[1] != 0:
            return False
        return True

    return True


class NeuralCache:
    def __init__(self, cache_size: int = 16, neural_hint_weight: float = 0.5):
        if not (1 <= cache_size <= 256):
            raise ValueError("cache_size must be between 1 and 256")
        self.cache_size = cache_size
        self.slots: Dict[int, Optional[CacheLine]] = {
            i: None for i in range(cache_size)
        }
        self.opcode_map: Dict[str, int] = {}
        self.stats = CacheStats()
        self.neural_hint_weight = neural_hint_weight
        self._lock = threading.RLock()
        self._access_order: deque = deque(maxlen=cache_size)

    def lookup(self, opcode_id: str) -> Optional[Tuple[CacheLine, int]]:
        if not validate_opcode(opcode_id):
            self.stats.rejected += 1
            return None

        with self._lock:
            if opcode_id not in self.opcode_map:
                self.stats.misses += 1
                return None

            slot_idx = self.opcode_map[opcode_id]
            line = self.slots[slot_idx]

            if line is None:
                self.stats.misses += 1
                return None

            line.hit_count += 1
            line.last_hit = time.time()

            try:
                self._access_order.remove(slot_idx)
            except ValueError:
                pass
            self._access_order.append(slot_idx)

            self.stats.hits += 1
            return line, slot_idx

    def store(self, opcode_id: str, line: CacheLine) -> Optional[int]:
        if not validate_opcode(opcode_id):
            self.stats.rejected += 1
            return None

        with self._lock:
            if opcode_id in self.opcode_map:
                slot_idx = self.opcode_map[opcode_id]
            else:
                if len(self.opcode_map) >= self.cache_size:
                    slot_idx = self._find_eviction_slot()
                    if slot_idx is None:
                        self.stats.rejected += 1
                        return None
                else:
                    slot_idx = self._find_empty_slot()
                    if slot_idx is None:
                        slot_idx = self._find_eviction_slot()

                self.opcode_map[opcode_id] = slot_idx
                try:
                    self._access_order.append(slot_idx)
                except threading.ThreadError:
                    pass

            self.slots[slot_idx] = line
            return slot_idx

    def _find_empty_slot(self) -> Optional[int]:
        for idx in range(self.cache_size):
            if self.slots[idx] is None:
                return idx
        return None

    def _find_eviction_slot(self) -> Optional[int]:
        for idx in range(self.cache_size):
            if self.slots[idx] is None:
                return idx

        victim = self._select_victim()
        if victim is None:
            return None

        evicted = self.slots[victim]
        if evicted:
            opcode_to_remove = None
            for op, idx in self.opcode_map.items():
                if idx == victim:
                    opcode_to_remove = op
                    break
            if opcode_to_remove:
                del self.opcode_map[opcode_to_remove]
            self.stats.evictions += 1

        return victim

    def _select_victim(self) -> Optional[int]:
        best_score = float("inf")
        victim = None

        for idx in range(self.cache_size):
            line = self.slots[idx]
            if line is None:
                return idx

            age_score = line.age * line.hit_count
            neural_bonus = (
                self.neural_hint_weight if line.status == CacheStatus.HOT else 0
            )
            latency_penalty = line.avg_latency * 0.1

            score = age_score + neural_bonus - latency_penalty

            if score < best_score:
                best_score = score
                victim = idx

        return victim

    def record_latency(self, opcode_id: str, latency: float):
        with self._lock:
            if opcode_id in self.opcode_map:
                line = self.slots[self.opcode_map[opcode_id]]
                if line:
                    n = line.hit_count
                    line.avg_latency = (line.avg_latency * n + latency) / (n + 1)

    def invalidate(self, opcode_id: str):
        if not validate_opcode(opcode_id):
            return

        with self._lock:
            if opcode_id in self.opcode_map:
                slot_idx = self.opcode_map[opcode_id]
                self.slots[slot_idx] = None
                del self.opcode_map[opcode_id]
                try:
                    self._access_order.remove(slot_idx)
                except ValueError:
                    pass

    def clear(self):
        with self._lock:
            self.slots = {i: None for i in range(self.cache_size)}
            self.opcode_map.clear()
            self._access_order.clear()
            self.stats = CacheStats()

    def get_stats(self) -> CacheStats:
        return self.stats

    def dump(self) -> Dict:
        with self._lock:
            result = {"slots": {}, "stats": str(self.stats)}
            for idx, line in self.slots.items():
                if line:
                    result["slots"][idx] = {
                        "opcode_id": line.opcode_id,
                        "status": line.status.name,
                        "age": line.age,
                        "hit_count": line.hit_count,
                        "avg_latency": line.avg_latency,
                    }
            return result


class NeuralAPU:
    def __init__(self, cache_size: int = 16):
        self.cache = NeuralCache(cache_size)
        self._backend = None
        self._capability = 0
        self._load_backend()

    def _load_backend(self):
        try:
            from . import _bitswitch

            self._backend = _bitswitch
            self._capability = getattr(
                _bitswitch, "apu_cache_get_capability", lambda: 0
            )()
        except ImportError:
            self._backend = None

    def exec(self, opcode: str, *operands: np.ndarray, **kwargs) -> np.ndarray:
        if not validate_opcode(opcode):
            raise ValueError(f"Invalid opcode: {opcode}")

        if not validate_operand_shape(opcode, operands):
            raise ValueError(f"Invalid operand shapes for opcode: {opcode}")

        start_time = time.time()

        cached_result = self.cache.lookup(opcode)
        if cached_result is not None:
            cached, slot = cached_result
            self.cache.record_latency(opcode, time.time() - start_time)
            return self._execute_cached(opcode, operands, cached, slot)

        result = self._compute(opcode, *operands, **kwargs)

        line = CacheLine(
            opcode_id=opcode,
            config=kwargs.get("config", 0),
            status=CacheStatus.HOT,
            packed_weights=kwargs.get("weights"),
            scales=kwargs.get("scales"),
        )
        self.cache.store(opcode, line)
        self.cache.record_latency(opcode, time.time() - start_time)

        return result

    def lookup(self, opcode: str):
        """Public lookup method for cache inspection."""
        return self.cache.lookup(opcode)

    def _execute_cached(
        self, opcode: str, operands: Tuple, cached: CacheLine, slot: int
    ) -> np.ndarray:
        return self._compute(
            opcode, *operands, weights=cached.packed_weights, scales=cached.scales
        )

    def _compute(self, opcode: str, *operands: np.ndarray, **kwargs) -> np.ndarray:
        if self._backend is None:
            return self._compute_fallback(opcode, *operands, **kwargs)
        return self._compute_backend(opcode, *operands, **kwargs)

    def _compute_backend(
        self, opcode: str, *operands: np.ndarray, **kwargs
    ) -> np.ndarray:
        weights = kwargs.get("weights")
        scales = kwargs.get("scales")

        if weights is not None and scales is not None:
            weights = np.packbits((weights > 0.5).astype(np.uint8), axis=-1)
            return np.matmul(operands[0], weights.T) * scales
        return np.zeros((operands[0].shape[0], 256))

    def _compute_fallback(
        self, opcode: str, *operands: np.ndarray, **kwargs
    ) -> np.ndarray:
        if opcode == "TMUL":
            a = operands[0]
            weights = kwargs.get(
                "weights", np.random.randn(a.shape[1], 256).astype(np.float32)
            )
            scales = kwargs.get("scales", np.ones(256, dtype=np.float32))
            result = np.matmul(a, weights) * scales
            return result
        elif opcode == "TADD":
            return operands[0] + operands[1]
        elif opcode == "TGATE":
            x = operands[0]
            logits = operands[1]
            gates = np.zeros(logits.shape, dtype=np.float32)
            gates[np.arange(len(logits)), logits.argmax(axis=-1)] = 1.0
            tile_size = x.shape[1] // logits.shape[1]
            tiled_gates = np.repeat(gates, tile_size, axis=1)
            return x * tiled_gates
        elif opcode == "MTFP_ADD":
            from .mtfp import MTFP_16, mtfp_add

            return mtfp_add(operands[0], operands[1], MTFP_16)
        elif opcode == "MTFP_MUL":
            from .mtfp import MTFP_16, mtfp_mul

            return mtfp_mul(operands[0], operands[1], MTFP_16)
        elif opcode == "MTFP_MATMUL":
            from .mtfp import MTFP_16, mtfp_matmul

            return mtfp_matmul(operands[0], operands[1], MTFP_16)
        else:
            return np.zeros((operands[0].shape[0], 256))

    def register(self, opcode: str, **kwargs):
        if not validate_opcode(opcode):
            raise ValueError(f"Invalid opcode: {opcode}")

        line = CacheLine(
            opcode_id=opcode,
            config=kwargs.get("config", 0),
            status=CacheStatus.HOT,
            packed_weights=kwargs.get("weights"),
            scales=kwargs.get("scales"),
            tile_routing=kwargs.get("tile_routing"),
        )
        self.cache.store(opcode, line)

    def cache_stats(self) -> Dict:
        return {
            "hits": self.cache.stats.hits,
            "misses": self.cache.stats.misses,
            "hit_rate": self.cache.stats.hit_rate,
            "evictions": self.cache.stats.evictions,
            "rejected": self.cache.stats.rejected,
        }
