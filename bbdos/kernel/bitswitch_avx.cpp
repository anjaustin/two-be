#include "bitswitch.h"
#include <cstring>
#include <cmath>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

extern "C" void bitswitch_linear_forward_avx(
    const float* input,
    const uint8_t* packed_w,
    const float* scales,
    const int8_t* gate_mask,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
);

#if USE_AVX

static inline float dot_packed_avx(
    const float* input,
    const uint8_t* packed_row,
    int in_features
) {
    __m256 acc_vec = _mm256_setzero_ps();
    int packed_len = (in_features + 3) / 4;
    int full_packs = in_features / 4;

    for (int p = 0; p < full_packs; p++) {
        uint8_t packed = packed_row[p];
        __m256 in_vec = _mm256_loadu_ps(&input[p * 4]);

        float weights[8];
        for (int i = 0; i < 4; i++) {
            uint8_t code = (packed >> (i * 2)) & 0x03;
            weights[i] = (code == 0x01) ? 1.0f : (code == 0x02) ? -1.0f : 0.0f;
        }
        __m128 w_lo = _mm_loadu_ps(weights);
        __m256 w_vec = _mm256_castps128ps256(w_lo);

        acc_vec = _mm256_fmadd_ps(in_vec, w_vec, acc_vec);
    }

    float acc[8];
    _mm256_storeu_ps(acc, acc_vec);
    float sum = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];

    int remaining_start = full_packs * 4;
    if (remaining_start < in_features) {
        uint8_t packed = packed_row[full_packs];
        for (int i = 0; remaining_start + i < in_features; i++) {
            uint8_t code = (packed >> (i * 2)) & 0x03;
            if (code == 0x01) {
                sum += input[remaining_start + i];
            } else if (code == 0x02) {
                sum -= input[remaining_start + i];
            }
        }
    }

    return sum;
}

static inline void dot_packed_avx_batch(
    const float* input,
    const uint8_t* packed_row,
    int in_features,
    float* output,
    int out_cols
) {
    int packed_len = (in_features + 3) / 4;
    int full_packs = in_features / 4;

    for (int o = 0; o < out_cols; o++) {
        __m256 acc_vec = _mm256_setzero_ps();
        const uint8_t* w_row = &packed_row[o * packed_len];

        for (int p = 0; p < full_packs; p++) {
            uint8_t packed = w_row[p];
            __m256 in_vec = _mm256_loadu_ps(&input[p * 4]);

            float weights[8] = {0};
            for (int i = 0; i < 4; i++) {
                uint8_t code = (packed >> (i * 2)) & 0x03;
                weights[i] = (code == 0x01) ? 1.0f : (code == 0x02) ? -1.0f : 0.0f;
            }
            __m128 w_lo = _mm_loadu_ps(weights);
            __m256 w_vec = _mm256_castps128ps256(w_lo);

            acc_vec = _mm256_fmadd_ps(in_vec, w_vec, acc_vec);
        }

        float acc[8];
        _mm256_storeu_ps(acc, acc_vec);
        float sum = 0;
        for (int i = 0; i < 8 && (full_packs * 4 + i) < in_features; i++) {
            sum += acc[i];
        }
        output[o] = sum;
    }
}

extern "C" void bitswitch_linear_forward_avx(
    const float* input,
    const uint8_t* packed_w,
    const float* scales,
    const int8_t* gate_mask,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
) {
    int out_per_tile = out_features / num_tiles;
    int packed_in_dim = (in_features + 3) / 4;

    memset(output, 0, batch_size * out_features * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        const float* in_row = &input[b * in_features];
        float* out_row = &output[b * out_features];

        for (int t = 0; t < num_tiles; t++) {
            if (gate_mask[b * num_tiles + t] == 0) {
                continue;
            }

            int out_start = t * out_per_tile;
            const uint8_t* tile_weights = &packed_w[t * out_per_tile * packed_in_dim];
            const float* tile_scales = &scales[t * out_per_tile];

            float tile_output[4096];
            dot_packed_avx_batch(in_row, tile_weights, in_features, tile_output, out_per_tile);

            for (int o = 0; o < out_per_tile; o++) {
                out_row[out_start + o] = tile_output[o] * tile_scales[o];
            }
        }
    }
}

#else

extern "C" void bitswitch_linear_forward_avx(
    const float* input,
    const uint8_t* packed_w,
    const float* scales,
    const int8_t* gate_mask,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
) {
    bitswitch_linear_forward(input, packed_w, scales, gate_mask, output,
                            batch_size, in_features, out_features, num_tiles);
}

#endif

extern "C" void apu_tmul(
    const float* a,
    const uint8_t* packed_b,
    const float* scales,
    float* output,
    int m, int k, int n
) {
    int packed_k = (k + 3) / 4;

    memset(output, 0, m * n * sizeof(float));

    for (int i = 0; i < m; i++) {
        const float* a_row = &a[i * k];
        float* out_row = &output[i * n];

        for (int j = 0; j < n; j++) {
            const uint8_t* b_col = &packed_b[j * packed_k];
            float acc = 0.0f;

            for (int p = 0; p < packed_k; p++) {
                uint8_t packed = b_col[p];
                int base = p * 4;

                for (int idx = 0; idx < 4 && base + idx < k; idx++) {
                    uint8_t code = (packed >> (idx * 2)) & 0x03;
                    if (code == 0x01) {
                        acc += a_row[base + idx];
                    } else if (code == 0x10) {
                        acc -= a_row[base + idx];
                    }
                }
            }
            out_row[j] = acc * scales[j];
        }
    }
}

extern "C" void apu_tadd(
    const float* a,
    const float* b,
    float* output,
    int size
) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

extern "C" void apu_tgate(
    const float* input,
    const float* gate_logits,
    int8_t* gate_mask,
    float* output,
    int batch, int channels, int num_tiles
) {
    int tile_size = channels / num_tiles;

    for (int b = 0; b < batch; b++) {
        int best_tile = 0;
        float best_score = gate_logits[b * num_tiles];

        for (int t = 1; t < num_tiles; t++) {
            float score = gate_logits[b * num_tiles + t];
            if (score > best_score) {
                best_score = score;
                best_tile = t;
            }
        }

        for (int t = 0; t < num_tiles; t++) {
            gate_mask[b * num_tiles + t] = (t == best_tile) ? 1 : 0;
        }

        const float* in_row = &input[b * channels];
        float* out_row = &output[b * channels];
        memset(out_row, 0, channels * sizeof(float));

        int out_start = best_tile * tile_size;
        memcpy(&out_row[out_start], &in_row[out_start], tile_size * sizeof(float));
    }
}

extern "C" uint64_t apu_cache_get_capability(void) {
#if USE_AVX
    return 0x00000001;
#elif USE_NEON
    return 0x00000002;
#else
    return 0x00000000;
#endif
}
