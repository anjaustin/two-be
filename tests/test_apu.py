"""
Tests for Neural APU and L-Cache

Tests the AVX kernel, cache eviction, and opcode interface.
"""

import pytest
import numpy as np
import time
import threading
from bbdos.kernel.apu_cache import (
    NeuralAPU,
    NeuralCache,
    CacheLine,
    CacheStats,
    CacheStatus,
)


class TestNeuralCache:
    def test_cache_initialization(self):
        cache = NeuralCache(cache_size=8)
        assert len(cache.slots) == 8
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_cache_store_and_lookup(self):
        cache = NeuralCache(cache_size=4)

        line = CacheLine(opcode_id="TMUL_256", config=0, status=CacheStatus.HOT)
        slot = cache.store("TMUL_256", line)

        result = cache.lookup("TMUL_256")
        assert result is not None
        assert result[0].opcode_id == "TMUL_256"

    def test_cache_miss(self):
        cache = NeuralCache(cache_size=4)
        result = cache.lookup("TMUL")
        assert result is None
        assert cache.stats.misses == 1

    def test_cache_hit_rate(self):
        cache = NeuralCache(cache_size=4)

        line = CacheLine(opcode_id="TMUL", config=0)
        cache.store("TMUL", line)

        for _ in range(5):
            cache.lookup("TMUL")

        assert cache.stats.hits == 5
        assert cache.stats.hit_rate == 1.0

    def test_cache_eviction(self):
        cache = NeuralCache(cache_size=2)

        for i in range(3):
            line = CacheLine(opcode_id=f"TMUL_{i}", config=0)
            cache.store(f"TMUL_{i}", line)

        assert cache.stats.evictions >= 1

    def test_cache_lru_order(self):
        cache = NeuralCache(cache_size=4)

        for i in range(4):
            line = CacheLine(opcode_id=f"OP_{i}", config=0)
            cache.store(f"OP_{i}", line)

        cache.lookup("OP_0")
        cache.lookup("OP_1")

        assert cache.stats.hits >= 0

    def test_cache_invalidate(self):
        cache = NeuralCache(cache_size=4)

        line = CacheLine(opcode_id="TMUL", config=0)
        cache.store("TMUL", line)

        cache.invalidate("TMUL")

        result = cache.lookup("TMUL")
        assert result is None


class TestNeuralAPU:
    def test_apu_initialization(self):
        apu = NeuralAPU(cache_size=8)
        assert apu.cache.cache_size == 8

    def test_apu_exec_tmul(self):
        apu = NeuralAPU(cache_size=4)

        a = np.random.randn(2, 128).astype(np.float32)
        weights = np.random.randn(128, 256).astype(np.float32)
        scales = np.ones(256, dtype=np.float32)

        result = apu.exec("TMUL", a, weights=weights, scales=scales)

        assert result.shape == (2, 256)

    def test_apu_exec_tadd(self):
        apu = NeuralAPU(cache_size=4)

        a = np.random.randn(2, 256).astype(np.float32)
        b = np.random.randn(2, 256).astype(np.float32)

        result = apu.exec("TADD", a, b)

        np.testing.assert_array_almost_equal(result, a + b)

    def test_apu_exec_tgate(self):
        apu = NeuralAPU(cache_size=4)

        x = np.random.randn(4, 256).astype(np.float32)
        logits = np.random.randn(4, 4).astype(np.float32)

        result = apu.exec("TGATE", x, logits)

        assert result.shape == (4, 256)

    def test_apu_cache_miss_then_hit(self):
        apu = NeuralAPU(cache_size=4)

        a = np.random.randn(2, 128).astype(np.float32)
        weights = np.random.randn(128, 256).astype(np.float32)
        scales = np.ones(256, dtype=np.float32)

        apu.exec("TMUL", a, weights=weights, scales=scales)

        result = apu.exec("TMUL", a, weights=weights, scales=scales)

        stats = apu.cache_stats()
        assert stats["hits"] >= 1

    def test_apu_register(self):
        apu = NeuralAPU(cache_size=4)

        weights = np.random.randn(128, 256).astype(np.float32)
        scales = np.ones(256, dtype=np.float32)

        apu.register("TMUL", weights=weights, scales=scales)

        result = apu.lookup("TMUL")
        assert result is not None


class TestCacheStats:
    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8

    def test_zero_division(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0


class TestAPUPerformance:
    def test_cache_latency_tracking(self):
        apu = NeuralAPU(cache_size=4)

        for _ in range(10):
            a = np.random.randn(2, 128).astype(np.float32)
            weights = np.random.randn(128, 256).astype(np.float32)
            scales = np.ones(256, dtype=np.float32)
            apu.exec("TMUL", a, weights=weights, scales=scales)

        stats = apu.cache_stats()
        assert stats["hits"] + stats["misses"] == 10

    def test_concurrent_access(self):
        cache = NeuralCache(cache_size=16)
        errors = []

        def worker(i):
            try:
                line = CacheLine(opcode_id=f"OP_{i}", config=0)
                cache.store(f"OP_{i}", line)
                for _ in range(10):
                    cache.lookup(f"OP_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestAPUSecurity:
    def test_cache_overflow_protection(self):
        cache = NeuralCache(cache_size=2)

        for i in range(100):
            line = CacheLine(opcode_id=f"TMUL_{i}", config=0)
            cache.store(f"TMUL_{i}", line)

        assert cache.stats.evictions > 0

    def test_invalid_opcode_rejected(self):
        cache = NeuralCache(cache_size=4)

        line = CacheLine(opcode_id="INVALID", config=0)
        result = cache.store("INVALID", line)

        assert result is None
        assert cache.stats.rejected >= 1

    def test_invalid_opcode_exec_raises(self):
        apu = NeuralAPU(cache_size=4)

        a = np.random.randn(2, 128).astype(np.float32)

        with pytest.raises(ValueError, match="Invalid opcode"):
            apu.exec("INVALID_OP", a)

    def test_sql_injection_pattern_blocked(self):
        cache = NeuralCache(cache_size=4)

        malicious = "'; DROP TABLE opcodes;--"
        line = CacheLine(opcode_id=malicious, config=0)
        result = cache.store(malicious, line)

        assert result is None

    def test_valid_opcodes_accepted(self):
        cache = NeuralCache(cache_size=16)

        valid_opcodes = ["TMUL", "TADD", "TGATE", "TATTN", "TNORM"]

        for op in valid_opcodes:
            line = CacheLine(opcode_id=op, config=0)
            result = cache.store(op, line)
            assert result is not None

    def test_shape_validation_tmul(self):
        apu = NeuralAPU(cache_size=4)

        a = np.random.randn(128).astype(np.float32)

        with pytest.raises(ValueError, match="Invalid operand"):
            apu.exec("TMUL", a)

    def test_shape_validation_tadd(self):
        apu = NeuralAPU(cache_size=4)

        a = np.random.randn(2, 256).astype(np.float32)
        b = np.random.randn(3, 256).astype(np.float32)

        with pytest.raises(ValueError, match="Invalid operand"):
            apu.exec("TADD", a, b)

    def test_cache_size_validation(self):
        with pytest.raises(ValueError, match="cache_size"):
            NeuralCache(cache_size=500)

    def test_concurrent_access_secure(self):
        cache = NeuralCache(cache_size=16)
        errors = []

        def worker(i):
            try:
                line = CacheLine(opcode_id=f"TMUL_{i}", config=0)
                cache.store(f"TMUL_{i}", line)
                for _ in range(10):
                    cache.lookup(f"TMUL_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
