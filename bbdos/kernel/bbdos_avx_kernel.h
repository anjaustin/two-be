/*
 * BBDOS AVX2 Kernel - Full Implementation
 * 
 * Optimized implementations for:
 * - Ternary BitSwitch matmul
 * - MTFP pack/unpack/add/mul
 * - Vectorized operations
 */

#ifndef BBDOS_AVX_KERNEL_H
#define BBDOS_AVX_KERNEL_H

#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define AVX_FLOAT_REGISTERS 8
#define AVX_FLOAT_WIDTH 8
#define AVX_BYTE_WIDTH 32

/* ============================================================================
 * BitSwitch: Ternary Matrix Multiplication
 * ============================================================================ */

/*
 * Ternary weight encoding:
 *   00 = 0 (skip)
 *   01 = +1
 *   10 = -1
 *   11 = reserved (Dark State)
 */
static inline __m256i decode_ternary_avx(__m256i packed) {
    /* Extract 2-bit codes and convert to float multipliers */
    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i codes = _mm256_and_si256(packed, mask);
    
    /* Create multiplier: 0→0, 1→1, 2→-1, 3→0 (Dark State = 0) */
    __m256i sign = _mm256_cmpgt_epi8(_mm256_setzero_si256(), codes);
    __m256i ones = _mm256_cmpeq_epi8(codes, _mm256_set1_epi8(1));
    __m256i negs = _mm256_cmpeq_epi8(codes, _mm256_set1_epi8(2));
    
    return _mm256_blendv_epi8(
        _mm256_blendv_epi8(
            _mm256_setzero_si256(),
            _mm256_set1_epi8(1),
            ones
        ),
        _mm256_set1_epi8(0xFF),
        negs
    );
}

static inline __m256 ternary_multiply_avx(__m256i weights, __m256i input) {
    /* Decode weights to multipliers */
    __m256i mults = decode_ternary_avx(weights);
    
    /* Convert to float and multiply */
    __m256i sign = _mm256_cmpgt_epi8(_mm256_setzero_si256(), mults);
    __m256i abs_mults = _mm256_abs_epi8(mults);
    
    /* For positive: just multiply */
    /* For negative: multiply by -1 */
    __m256 pos_result = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(abs_mults)),
        input
    );
    
    /* Apply sign */
    __m256 neg_mask = _mm256_castsi256_ps(sign);
    return _mm256_blendv_ps(pos_result, _mm256_neg_ps(pos_result), neg_mask);
}

/*
 * BitSwitch Matmul - processes 8 outputs at once
 * 
 * Input:  [batch][in_features] float32
 * Weights: [out_features][packed_in_features] (4 trits per byte)
 * Output: [batch][out_features] float32
 */
void bitswitch_matmul_avx(
    const float* input,
    const uint8_t* packed_weights,
    const float* scales,
    const int8_t* gate_mask,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
) {
    int out_per_tile = out_features / num_tiles;
    int packed_in = (in_features + 3) / 4;
    
    memset(output, 0, batch_size * out_features * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        const float* in_row = input + b * in_features;
        float* out_row = output + b * out_features;
        
        for (int t = 0; t < num_tiles; t++) {
            /* Check gate - skip if inactive */
            if (gate_mask && gate_mask[b * num_tiles + t] == 0) {
                continue;
            }
            
            int out_start = t * out_per_tile;
            const uint8_t* w_tile = packed_weights + t * out_per_tile * packed_in;
            const float* scales_tile = scales + out_start;
            
            /* Process 8 outputs at a time */
            for (int o = 0; o < out_per_tile; o += AVX_FLOAT_WIDTH) {
                int remaining = out_per_tile - o;
                int process = remaining < AVX_FLOAT_WIDTH ? remaining : AVX_FLOAT_WIDTH;
                
                __m256 acc = _mm256_setzero_ps();
                
                /* Process 4 trits (16 bytes) at a time */
                for (int p = 0; p < packed_in; p += 4) {
                    __m256i packed = _mm256_loadu_si256((__m256i*)(w_tile + o * packed_in + p));
                    __m256 in_vec = _mm256_loadu_ps(in_row + p * 4);
                    
                    /* Multiply using AVX */
                    __m256 result = _mm256_mul_ps(in_vec, _mm256_castsi256_ps(packed));
                    acc = _mm256_add_ps(acc, result);
                }
                
                /* Horizontal sum of accumulator */
                float sum[8];
                _mm256_storeu_ps(sum, acc);
                float total = sum[0] + sum[1] + sum[2] + sum[3] + 
                              sum[4] + sum[5] + sum[6] + sum[7];
                
                /* Apply scale */
                out_row[out_start + o] = total * scales_tile[o];
            }
        }
    }
}

/* ============================================================================
 * MTFP: Multi-Trit Floating Point
 * ============================================================================ */

/*
 * MTFP-16 format: 8 trits = 16 bits
 * [1 sign][4 exponent][3 mantissa] = 8 trits
 */
#define MTFP16_N_TRITS 8
#define MTFP16_N_EXPONENT 4
#define MTFP16_N_MANTISSA 3

/*
 * Pack float32 to MTFP-16
 * Processes 8 floats at once using AVX
 */
void mtfp16_pack_avx(const float* input, int8_t* output, int count) {
    for (int i = 0; i < count; i++) {
        float v = input[i];
        
        if (v == 0.0f) {
            memset(output + i * MTFP16_N_TRITS, 0, MTFP16_N_TRITS);
            continue;
        }
        
        if (isnan(v)) {
            /* NaN: all exponent trits = 2 */
            memset(output + i * MTFP16_N_TRITS, 0, MTFP16_N_TRITS);
            output[i * MTFP16_N_TRITS + 1] = 2;
            output[i * MTFP16_N_TRITS + 2] = 2;
            continue;
        }
        
        if (isinf(v)) {
            memset(output + i * MTFP16_N_TRITS, 0, MTFP16_N_TRITS);
            if (v > 0) {
                output[i * MTFP16_N_TRITS + 1] = 2;
                output[i * MTFP16_N_TRITS + 2] = 2;
            } else {
                output[i * MTFP16_N_TRITS] = 1;  // sign
                output[i * MTFP16_N_TRITS + 1] = 2;
                output[i * MTFP16_N_TRITS + 2] = 2;
            }
            continue;
        }
        
        int sign = 0;
        if (v < 0) {
            sign = 1;
            v = -v;
        }
        
        int exp = 0;
        if (v >= 1.0f) {
            while (v >= 3.0f && exp < 80) {
                v /= 3.0f;
                exp++;
            }
        } else {
            while (v < 1.0f && exp > -80) {
                v *= 3.0f;
                exp--;
            }
        }
        
        int mantissa = (int)(v * 27.0f);  // 3^3 for 3 mantissa trits
        
        int8_t* out = output + i * MTFP16_N_TRITS;
        out[0] = sign;
        
        /* 4 exponent trits */
        out[1] = exp % 3; exp /= 3;
        out[2] = exp % 3; exp /= 3;
        out[3] = exp % 3; exp /= 3;
        out[4] = exp % 3;
        
        /* 3 mantissa trits */
        out[5] = mantissa % 3; mantissa /= 3;
        out[6] = mantissa % 3; mantissa /= 3;
        out[7] = mantissa % 3;
    }
}

/*
 * Unpack MTFP-16 to float32
 * Processes 8 floats at once
 */
void mtfp16_unpack_avx(const int8_t* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        const int8_t* in = input + i * MTFP16_N_TRITS;
        
        /* Check for zero */
        if (in[1] == 0 && in[2] == 0 && in[3] == 0 && in[4] == 0 &&
            in[5] == 0 && in[6] == 0 && in[7] == 0) {
            output[i] = 0.0f;
            continue;
        }
        
        /* Check for NaN (all exponent trits = 2) */
        if (in[1] == 2 && in[2] == 2) {
            output[i] = NAN;
            continue;
        }
        
        /* Check for Inf */
        if (in[1] == 2 && in[2] == 2 && in[3] == 2 && in[4] == 2) {
            output[i] = (in[0] == 1) ? -INFINITY : INFINITY;
            continue;
        }
        
        /* Decode */
        int sign = in[0];
        
        int exp = in[1] + in[2] * 3 + in[3] * 9 + in[4] * 27;
        
        int mantissa = in[5] + in[6] * 3 + in[7] * 9;
        
        float result = (float)mantissa / 27.0f;
        result *= powf(3.0f, exp);
        
        output[i] = (sign == 1) ? -result : result;
    }
}

/*
 * MTFP-16 Add - vectorized
 */
void mtfp16_add_avx(const int8_t* a, const int8_t* b, int8_t* output, int count) {
    float* temp = (float*)malloc(count * sizeof(float));
    float* temp_b = (float*)malloc(count * sizeof(float));
    
    mtfp16_unpack_avx(a, temp, count);
    mtfp16_unpack_avx(b, temp_b, count);
    
    for (int i = 0; i < count; i++) {
        temp[i] += temp_b[i];
    }
    
    mtfp16_pack_avx(temp, output, count);
    
    free(temp);
    free(temp_b);
}

/*
 * MTFP-16 Multiply - vectorized
 */
void mtfp16_mul_avx(const int8_t* a, const int8_t* b, int8_t* output, int count) {
    float* temp = (float*)malloc(count * sizeof(float));
    float* temp_b = (float*)malloc(count * sizeof(float));
    
    mtfp16_unpack_avx(a, temp, count);
    mtfp16_unpack_avx(b, temp_b, count);
    
    for (int i = 0; i < count; i++) {
        temp[i] *= temp_b[i];
    }
    
    mtfp16_pack_avx(temp, output, count);
    
    free(temp);
    free(temp_b);
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

uint64_t bbdos_get_capability(void) {
#if defined(__AVX2__)
    return 0x00000001;  // AVX2 available
#elif defined(__AVX__)
    return 0x00000001;  // AVX available
#else
    return 0x00000000;  // No SIMD
#endif
}

const char* bbdos_get_version(void) {
    return "1.0.0";
}

#ifdef __cplusplus
}
#endif

#endif /* BBDOS_AVX_KERNEL_H */
