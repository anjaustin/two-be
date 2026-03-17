# BBDOS Kernel Documentation

## Overview

The BBDOS kernel is implemented in pure C with AVX2 acceleration. All compute-heavy operations are offloaded to C for maximum performance.

## Build

```bash
# Build libbbdos_apu.so
gcc -O3 -mavx2 -march=skylake -fPIC -shared -fopenmp \
    bbdos/kernel/bbdos_apu.c -o bbdos/kernel/libbbdos_apx.so -lm -lpthread

# Build libbbdos_avx.so
gcc -O3 -mavx2 -march=skylake -fPIC -shared -fopenmp \
    bbdos/kernel/bbdos_avx.c -o bbdos/kernel/libbbdos_avx.so -lm -lpthread
```

## API Reference

### BBDOS_APU Context

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef void BBDOS_APU;

// Create/destroy context
BBDOS_APU* bbdos_apu_create(int cache_size);
void bbdos_apu_destroy(BBDOS_APU* apu);

// Execute opcode
int bbdos_apu_exec(BBDOS_APU* apu, const char* opcode, 
                   void** operands, int* shapes, void* output);

// Get cache statistics
void bbdos_apu_stats(BBDOS_APU* apu, int* hits, int* misses, 
                     int* evictions, int* rejected);

// Capability query
uint64_t bbdos_capability(void);  // Returns 1 if AVX2 available
const char* bbdos_version(void);   // Returns version string
```

### BitSwitch Ternary Matmul

```c
void bitswitch_matmul_avx(
    const float* input,        // [batch][in_features]
    const uint8_t* packed_weights,  // [out_features][packed_in]
    const float* scales,       // [out_features]
    const int8_t* gate_mask,    // [batch][num_tiles] or NULL
    float* output,             // [batch][out_features]
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
);
```

**Weight Encoding:** 4 trits per byte (2 bits each)
- `0b00` → 0
- `0b01` → +1
- `0b10` → -1
- `0b11` → Dark State (reserved)

### MTFP Operations

```c
void mtfp16_pack_avx(const float* input, int8_t* output, int count);
void mtfp16_unpack_avx(const int8_t* input, float* output, int count);
void mtfp16_add_avx(const int8_t* a, const int8_t* b, int8_t* output, int count);
void mtfp16_mul_avx(const int8_t* a, const int8_t* b, int8_t* output, int count);
```

## Python Bindings

```python
from bbdos.kernel import BBDOS_APU, bitswitch_matmul_avx, capability, version

# Create APU context
apu = BBDOS_APU(cache_size=256)

# Execute MTFP operations
a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
result = apu.exec('MTFP_ADD', [a, b], [4])

# Get cache stats
stats = apu.stats()
# {'hits': 0, 'misses': 1, 'evictions': 0, 'rejected': 0}

# BitSwitch matmul
input_data = np.random.randn(32, 512).astype(np.float32)
weights = np.random.randint(0, 64, (256 * 128)).astype(np.uint8)
scales = np.random.randn(256).astype(np.float32)
gate = np.random.randint(0, 2, (32, 16)).astype(np.int8)
output = np.zeros((32, 256), dtype=np.float32)

bitswitch_matmul_avx(input_data, weights, scales, gate, output,
                     32, 512, 256, 16)
```

## Security

All C functions include:
- Null pointer guards
- Count validation (0 to 1M limit)
- Division-by-zero protection
- malloc → calloc (zero-initialization)

## Performance

| Operation | Throughput |
|-----------|-------------|
| MTFP pack/unpack | 85M ops/sec |
| MTFP add/mul | 40M ops/sec |
| BitSwitch matmul | 6K ops/sec |

## Version History

| Version | Notes |
|---------|-------|
| 1.0.0 | Initial Python implementation |
| 1.1.0 | C/AVX implementation (current) |
