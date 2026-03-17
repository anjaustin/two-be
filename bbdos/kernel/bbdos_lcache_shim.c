/*
 * BBDOS L-Cache Shim for BitNet.cpp Integration
 * 
 * Provides a bridge between BitNet.cpp and our L-Cache architecture.
 * Allows BitNet.cpp to leverage:
 * - L-Cache for attention caching
 * - MTFP for memory-efficient computation
 * - APU routing for dynamic kernel selection
 *
 * Build: gcc -O3 -mavx2 -march=skylake -fPIC -shared -fopenmp bbdos_lcache_shim.c -o libbbdos_lcache_shim.so -lm -lpthread ../bbdos_apu.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define MAX_CACHE_SIZE 512
#define MAX_OPCODE_LEN 32
#define MTFP16_TRITS 8

/* ============================================================================
 * FNV-1a Hash
 * ============================================================================ */

static uint64_t hash_data(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

/* ============================================================================
 * L-Cache Implementation
 * ============================================================================ */

typedef struct {
    char opcode[MAX_OPCODE_LEN];
    uint64_t data_hash;
    void* data;
    size_t data_size;
    double last_hit;
    int hit_count;
} CacheLine;

typedef struct {
    CacheLine slots[MAX_CACHE_SIZE];
    int cache_size;
    int hits;
    int misses;
    int evictions;
    pthread_mutex_t lock;
} BBDOS_LCache;

static BBDOS_LCache* cache_create(int cache_size) {
    if (cache_size > MAX_CACHE_SIZE) cache_size = MAX_CACHE_SIZE;
    if (cache_size < 1) cache_size = 64;
    
    BBDOS_LCache* c = (BBDOS_LCache*)calloc(1, sizeof(BBDOS_LCache));
    c->cache_size = cache_size;
    pthread_mutex_init(&c->lock, NULL);
    return c;
}

static void cache_destroy(BBDOS_LCache* c) {
    if (!c) return;
    for (int i = 0; i < c->cache_size; i++) {
        if (c->slots[i].data) free(c->slots[i].data);
    }
    pthread_mutex_destroy(&c->lock);
    free(c);
}

static int cache_lookup(BBDOS_LCache* c, const char* opcode, uint64_t data_hash) {
    pthread_mutex_lock(&c->lock);
    
    for (int i = 0; i < c->cache_size; i++) {
        if (c->slots[i].opcode[0] && 
            strcmp(c->slots[i].opcode, opcode) == 0 &&
            c->slots[i].data_hash == data_hash) {
            c->slots[i].hit_count++;
            c->slots[i].last_hit = (double)clock() / CLOCKS_PER_SEC;
            c->hits++;
            pthread_mutex_unlock(&c->lock);
            return i;
        }
    }
    
    c->misses++;
    pthread_mutex_unlock(&c->lock);
    return -1;
}

static int cache_store(BBDOS_LCache* c, const char* opcode, uint64_t data_hash, void* data, size_t size) {
    pthread_mutex_lock(&c->lock);
    
    for (int i = 0; i < c->cache_size; i++) {
        if (c->slots[i].opcode[0] && 
            strcmp(c->slots[i].opcode, opcode) == 0 &&
            c->slots[i].data_hash == data_hash) {
            if (c->slots[i].data) free(c->slots[i].data);
            c->slots[i].data = malloc(size);
            memcpy(c->slots[i].data, data, size);
            c->slots[i].data_size = size;
            pthread_mutex_unlock(&c->lock);
            return i;
        }
    }
    
    for (int i = 0; i < c->cache_size; i++) {
        if (c->slots[i].opcode[0] == 0) {
            strncpy(c->slots[i].opcode, opcode, MAX_OPCODE_LEN - 1);
            c->slots[i].data_hash = data_hash;
            c->slots[i].data = malloc(size);
            memcpy(c->slots[i].data, data, size);
            c->slots[i].data_size = size;
            c->slots[i].last_hit = (double)clock() / CLOCKS_PER_SEC;
            pthread_mutex_unlock(&c->lock);
            return i;
        }
    }
    
    int lru_idx = 0;
    double oldest = 1e100;
    for (int i = 0; i < c->cache_size; i++) {
        if (c->slots[i].last_hit < oldest) {
            oldest = c->slots[i].last_hit;
            lru_idx = i;
        }
    }
    
    free(c->slots[lru_idx].data);
    strncpy(c->slots[lru_idx].opcode, opcode, MAX_OPCODE_LEN - 1);
    c->slots[lru_idx].data_hash = data_hash;
    c->slots[lru_idx].data = malloc(size);
    memcpy(c->slots[lru_idx].data, data, size);
    c->slots[lru_idx].data_size = size;
    c->evictions++;
    
    pthread_mutex_unlock(&c->lock);
    return lru_idx;
}

/* ============================================================================
 * MTFP - Multi-Trit Floating Point
 * ============================================================================ */

static inline void mtfp16_pack_scalar(float v, int8_t* out) {
    memset(out, 0, MTFP16_TRITS);
    if (v == 0.0f) return;
    if (isnan(v)) { out[1]=2; out[2]=2; return; }
    if (isinf(v)) { out[1]=2; out[2]=2; if (v < 0) out[0]=1; return; }
    
    int sign = (v < 0) ? 1 : 0;
    if (v < 0) v = -v;
    
    int exp = 0;
    if (v >= 1.0f) {
        while (v >= 3.0f && exp < 80) { v /= 3.0f; exp++; }
    } else {
        while (v < 1.0f && exp > -80) { v *= 3.0f; exp--; }
    }
    
    int mantissa = (int)(v * 27.0f);
    
    out[0] = sign;
    out[1] = exp % 3; exp /= 3;
    out[2] = exp % 3; exp /= 3;
    out[3] = exp % 3; exp /= 3;
    out[4] = exp % 3;
    out[5] = mantissa % 3; mantissa /= 3;
    out[6] = mantissa % 3; mantissa /= 3;
    out[7] = mantissa % 3;
}

static inline float mtfp16_unpack_scalar(const int8_t* in) {
    if (in[1]==0 && in[2]==0 && in[3]==0 && in[4]==0 && in[5]==0 && in[6]==0 && in[7]==0) return 0.0f;
    if (in[1]==2 && in[2]==2) return NAN;
    if (in[1]==2 && in[2]==2 && in[3]==2 && in[4]==2) return (in[0]==1) ? -INFINITY : INFINITY;
    
    int sign = in[0];
    int exp = in[1] + in[2]*3 + in[3]*9 + in[4]*27;
    int mantissa = in[5] + in[6]*3 + in[7]*9;
    
    float result = (float)mantissa / 27.0f * powf(3.0f, exp);
    return (sign == 1) ? -result : result;
}

/* ============================================================================
 * BitSwitch Ternary Matmul (Simplified)
 * ============================================================================ */

static inline float decode_trit(uint8_t code) {
    if (code == 0x01) return 1.0f;
    if (code == 0x10) return -1.0f;
    return 0.0f;
}

static void bitswitch_matmul(
    const float* input,
    const uint8_t* packed_weights,
    const float* scales,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int packed_in = (in_features + 3) / 4;
    
    memset(output, 0, batch_size * out_features * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        const float* in_row = input + b * in_features;
        float* out_row = output + b * out_features;
        
        for (int o = 0; o < out_features; o++) {
            float acc = 0.0f;
            
            for (int p = 0; p < packed_in; p++) {
                uint8_t packed = packed_weights[o * packed_in + p];
                
                for (int k = 0; k < 4 && (p * 4 + k) < in_features; k++) {
                    uint8_t code = (packed >> (k * 2)) & 0x03;
                    acc += decode_trit(code) * in_row[p * 4 + k];
                }
            }
            
            out_row[o] = acc * scales[o];
        }
    }
}

/* ============================================================================
 * Public API for BitNet.cpp Integration
 * ============================================================================ */

typedef struct {
    BBDOS_LCache* cache;
    int capability;
    char version[32];
} BBDOS_Context;

BBDOS_Context* bbdos_create(int cache_size) {
    BBDOS_Context* ctx = (BBDOS_Context*)calloc(1, sizeof(BBDOS_Context));
    ctx->cache = cache_create(cache_size);
    ctx->capability = HAS_AVX2 ? 1 : 0;
    snprintf(ctx->version, sizeof(ctx->version), "1.0.0-shim");
    return ctx;
}

void bbdos_destroy(BBDOS_Context* ctx) {
    if (ctx) {
        cache_destroy(ctx->cache);
        free(ctx);
    }
}

uint64_t bbdos_capability(void) {
    return HAS_AVX2 ? 1 : 0;
}

const char* bbdos_version(void) {
    return "1.0.0-bbdos-lcache-shim";
}

/* Attention Cache API */
int bbdos_cache_attention(
    BBDOS_Context* ctx,
    const float* q, const float* k, const float* v,
    int seq_len, int head_dim,
    float* output
) {
    if (!ctx || !q || !k || !v || !output) return -1;
    if (seq_len <= 0 || head_dim <= 0) return -1;
    
    /* Create cache key from Q input */
    size_t input_size = seq_len * head_dim * sizeof(float);
    uint64_t cache_key = hash_data(q, input_size);
    cache_key ^= hash_data(k, input_size);
    cache_key ^= hash_data(v, input_size);
    
    char opcode[64];
    snprintf(opcode, sizeof(opcode), "attn_%dx%d", seq_len, head_dim);
    
    /* Check cache */
    int slot = cache_lookup(ctx->cache, opcode, cache_key);
    if (slot >= 0) {
        memcpy(output, ctx->cache->slots[slot].data, seq_len * head_dim * sizeof(float));
        return 1; /* Cache hit */
    }
    
    /* Compute attention */
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for (int s = 0; s < seq_len; s++) {
        float max_att = -1e30f;
        for (int s2 = 0; s2 <= s; s2++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[s * head_dim + d] * k[s2 * head_dim + d];
            }
            score *= scale;
            if (score > max_att) max_att = score;
        }
        
        float sum_exp = 0.0f;
        for (int s2 = 0; s2 <= s; s2++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[s * head_dim + d] * k[s2 * head_dim + d];
            }
            score *= scale;
            sum_exp += expf(score - max_att);
        }
        
        for (int d = 0; d < head_dim; d++) {
            float weighted = 0.0f;
            for (int s2 = 0; s2 <= s; s2++) {
                float score = 0.0f;
                for (int dd = 0; dd < head_dim; dd++) {
                    score += q[s * head_dim + dd] * k[s2 * head_dim + dd];
                }
                score *= scale;
                float attn_weight = expf(score - max_att) / sum_exp;
                weighted += attn_weight * v[s2 * head_dim + d];
            }
            output[s * head_dim + d] = weighted;
        }
    }
    
    /* Store in cache */
    cache_store(ctx->cache, opcode, cache_key, output, seq_len * head_dim * sizeof(float));
    
    return 0; /* Cache miss - computed */
}

/* FFN Cache API */
int bbdos_cache_ffn(
    BBDOS_Context* ctx,
    const float* input,
    const uint8_t* weights,
    const float* scales,
    int batch_size, int in_feat, int out_feat,
    float* output
) {
    if (!ctx || !input || !weights || !output) return -1;
    if (batch_size <= 0 || in_feat <= 0 || out_feat <= 0) return -1;
    
    size_t input_size = batch_size * in_feat * sizeof(float);
    uint64_t cache_key = hash_data(input, input_size);
    cache_key ^= hash_data(weights, (in_feat * out_feat + 3) / 4);
    
    char opcode[64];
    snprintf(opcode, sizeof(opcode), "ffn_%dx%dx%d", batch_size, in_feat, out_feat);
    
    int slot = cache_lookup(ctx->cache, opcode, cache_key);
    if (slot >= 0) {
        memcpy(output, ctx->cache->slots[slot].data, batch_size * out_feat * sizeof(float));
        return 1;
    }
    
    bitswitch_matmul(input, weights, scales, output, batch_size, in_feat, out_feat);
    
    cache_store(ctx->cache, opcode, cache_key, output, batch_size * out_feat * sizeof(float));
    
    return 0;
}

/* Stats API */
void bbdos_stats(BBDOS_Context* ctx, int* hits, int* misses, int* evictions) {
    if (!ctx || !ctx->cache) return;
    pthread_mutex_lock(&ctx->cache->lock);
    if (hits) *hits = ctx->cache->hits;
    if (misses) *misses = ctx->cache->misses;
    if (evictions) *evictions = ctx->cache->evictions;
    pthread_mutex_unlock(&ctx->cache->lock);
}

/* MTFP Compression API */
void bbdos_mtfp_pack(const float* input, int8_t* output, int count) {
    for (int i = 0; i < count; i++) {
        mtfp16_pack_scalar(input[i], output + i * MTFP16_TRITS);
    }
}

void bbdos_mtfp_unpack(const int8_t* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        output[i] = mtfp16_unpack_scalar(input + i * MTFP16_TRITS);
    }
}
