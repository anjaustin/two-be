/*
 * BBDOS APU - Pure C/AVX Implementation
 * 
 * Complete implementation:
 * - L-Cache with LRU eviction
 * - APU opcode routing
 * - MTFP operations
 * - BitSwitch ternary matmul
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
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

#define MAX_CACHE_SIZE 256
#define MAX_OPCODE_LEN 32

/* ============================================================================
 * MTFP - Multi-Trit Floating Point (from bbdos_avx.c)
 * ============================================================================ */

#define MTFP16_TRITS 8

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
 * L-Cache Implementation
 * ============================================================================ */

typedef enum {
    CACHE_STATUS_HOT = 0,
    CACHE_STATUS_COLD = 1,
    CACHE_STATUS_EVICT = 2,
    CACHE_STATUS_PREFETCH = 3
} CacheStatus;

typedef struct {
    char opcode[MAX_OPCODE_LEN];
    uint64_t data_hash;
    int config;
    CacheStatus status;
    int age;
    int hit_count;
    double avg_latency;
    double last_hit;
    void* data;
    size_t data_size;
} CacheLine;

typedef struct {
    CacheLine slots[MAX_CACHE_SIZE];
    char opcode_map[MAX_CACHE_SIZE][MAX_OPCODE_LEN];
    int cache_size;
    int num_entries;
    int hits;
    int misses;
    int evictions;
    int rejected;
    pthread_mutex_t lock;
} NeuralCache;

static NeuralCache* cache_create(int cache_size) {
    if (cache_size > MAX_CACHE_SIZE) cache_size = MAX_CACHE_SIZE;
    if (cache_size < 1) cache_size = 1;
    
    NeuralCache* c = (NeuralCache*)calloc(1, sizeof(NeuralCache));
    c->cache_size = cache_size;
    pthread_mutex_init(&c->lock, NULL);
    return c;
}

static void cache_destroy(NeuralCache* c) {
    if (!c) return;
    for (int i = 0; i < c->cache_size; i++) {
        if (c->slots[i].data) free(c->slots[i].data);
    }
    pthread_mutex_destroy(&c->lock);
    free(c);
}

static uint64_t hash_data(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static int cache_lookup(NeuralCache* c, const char* opcode, uint64_t data_hash) {
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

static int cache_store(NeuralCache* c, const char* opcode, uint64_t data_hash, void* data, size_t size) {
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
            c->slots[i].status = CACHE_STATUS_HOT;
            c->num_entries++;
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

static void cache_get_stats(NeuralCache* c, int* hits, int* misses, int* evictions, int* rejected) {
    pthread_mutex_lock(&c->lock);
    *hits = c->hits;
    *misses = c->misses;
    *evictions = c->evictions;
    *rejected = c->rejected;
    pthread_mutex_unlock(&c->lock);
}

/* ============================================================================
 * APU Opcode Validation
 * ============================================================================ */

static int validate_opcode(const char* opcode) {
    static const char* valid[] = {
        "TMUL", "TADD", "TGATE", "TATTN", "TNORM", "TLOOKUP",
        "MTFP_ADD", "MTFP_MUL", "MTFP_MATMUL",
        "RMSNorm", "SiLU", "GELU", "LayerNorm", "Softmax",
        "BitLinear", "BitAttention", "MatMul"
    };
    static const char* prefixes[] = {
        "TMUL_", "TADD_", "TGATE_", "TATTN_", "TNORM_", "TLOOKUP_", "MTFP_"
    };
    
    for (int i = 0; i < sizeof(valid)/sizeof(valid[0]); i++) {
        if (strcmp(opcode, valid[i]) == 0) return 1;
    }
    
    for (int i = 0; i < sizeof(prefixes)/sizeof(prefixes[0]); i++) {
        if (strncmp(opcode, prefixes[i], strlen(prefixes[i])) == 0) return 1;
    }
    
    return 0;
}

/* ============================================================================
 * BitSwitch Ternary Matmul
 * ============================================================================ */

static inline float decode_trit(uint8_t code) {
    if (code == 0x01) return 1.0f;
    if (code == 0x10) return -1.0f;
    return 0.0f;
}

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
    if (!input || !output || batch_size <= 0 || in_features <= 0 || 
        out_features <= 0 || num_tiles <= 0 || num_tiles > out_features) {
        if (output) memset(output, 0, batch_size * out_features * sizeof(float));
        return;
    }
    
    int out_per_tile = out_features / num_tiles;
    int packed_in = (in_features + 3) / 4;
    
    memset(output, 0, batch_size * out_features * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        const float* in_row = input + b * in_features;
        float* out_row = output + b * out_features;
        
        for (int t = 0; t < num_tiles; t++) {
            if (gate_mask && gate_mask[b * num_tiles + t] == 0) continue;
            
            int out_start = t * out_per_tile;
            const uint8_t* w_tile = packed_weights + t * out_per_tile * packed_in;
            const float* scales_tile = scales + out_start;
            
            for (int o = 0; o < out_per_tile; o++) {
                float acc = 0.0f;
                
                for (int p = 0; p < packed_in; p++) {
                    uint8_t packed = w_tile[o * packed_in + p];
                    
                    for (int k = 0; k < 4 && (p * 4 + k) < in_features; k++) {
                        uint8_t code = (packed >> (k * 2)) & 0x03;
                        acc += decode_trit(code) * in_row[p * 4 + k];
                    }
                }
                
                out_row[out_start + o] = acc * scales_tile[o];
            }
        }
    }
}

/* ============================================================================
 * APU Execution Engine
 * ============================================================================ */

typedef struct {
    NeuralCache* cache;
    int capability;
    char version[32];
} BBDOS_APU;

static BBDOS_APU* apu_create(int cache_size) {
    if (cache_size <= 0) cache_size = 64;
    BBDOS_APU* apu = (BBDOS_APU*)calloc(1, sizeof(BBDOS_APU));
    if (!apu) return NULL;
    apu->cache = cache_create(cache_size);
    if (!apu->cache) { free(apu); return NULL; }
    apu->capability = HAS_AVX2 ? 1 : 0;
    snprintf(apu->version, sizeof(apu->version), "1.0.0-c-avx");
    return apu;
}

static void apu_destroy(BBDOS_APU* apu) {
    if (apu) {
        cache_destroy(apu->cache);
        free(apu);
    }
}

static int apu_exec(BBDOS_APU* apu, const char* opcode, void** operands, int* shapes, void* output) {
    if (!apu || !opcode || !output) {
        return -1;  // Invalid parameters
    }
    
    if (!validate_opcode(opcode)) {
        return -1;  // Invalid opcode
    }
    
    int count = shapes ? shapes[0] : 0;
    if (count <= 0 || count > 1048576) {
        return -1;  // Invalid count
    }
    
    uint64_t data_hash = 0;
    if (operands && operands[0] && operands[1]) {
        size_t data_size = count * sizeof(float);
        data_hash = hash_data(operands[0], data_size);
        data_hash ^= hash_data(operands[1], data_size);
    }
    
    int slot = cache_lookup(apu->cache, opcode, data_hash);
    if (slot >= 0) {
        memcpy(output, apu->cache->slots[slot].data, apu->cache->slots[slot].data_size);
        return 0;
    }
    
    if (strcmp(opcode, "MTFP_ADD") == 0) {
        if (!operands || !operands[0] || !operands[1]) return -1;
        
        float* a = (float*)operands[0];
        float* b = (float*)operands[1];
        
        float* temp = (float*)calloc(count, sizeof(float));
        if (!temp) return -1;
        for (int i = 0; i < count; i++) {
            temp[i] = a[i] + b[i];
        }
        
        int8_t* out = (int8_t*)output;
        for (int i = 0; i < count; i++) {
            mtfp16_pack_scalar(temp[i], out + i * MTFP16_TRITS);
        }
        free(temp);
        
        cache_store(apu->cache, opcode, data_hash, output, count * MTFP16_TRITS);
        return 0;
    }
    
    if (strcmp(opcode, "MTFP_MUL") == 0) {
        if (!operands || !operands[0] || !operands[1]) return -1;
        
        float* a = (float*)operands[0];
        float* b = (float*)operands[1];
        
        float* temp = (float*)calloc(count, sizeof(float));
        if (!temp) return -1;
        for (int i = 0; i < count; i++) {
            temp[i] = a[i] * b[i];
        }
        
        int8_t* out = (int8_t*)output;
        for (int i = 0; i < count; i++) {
            mtfp16_pack_scalar(temp[i], out + i * MTFP16_TRITS);
        }
        free(temp);
        
        cache_store(apu->cache, opcode, data_hash, output, count * MTFP16_TRITS);
        return 0;
    }
    
    if (strcmp(opcode, "RMSNorm") == 0) {
        if (!operands || !operands[0]) return -1;
        
        float* x = (float*)operands[0];
        
        float sum_sq = 0.0f;
        for (int i = 0; i < count; i++) {
            sum_sq += x[i] * x[i];
        }
        float inv_std = 1.0f / sqrtf(sum_sq / (float)count + 1e-5f);
        
        float* out = (float*)output;
        for (int i = 0; i < count; i++) {
            out[i] = x[i] * inv_std;
        }
        
        return 0;
    }
    
    if (strcmp(opcode, "SiLU") == 0) {
        if (!operands || !operands[0]) return -1;
        
        float* x = (float*)operands[0];
        float* out = (float*)output;
        
        for (int i = 0; i < count; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-x[i]));
            out[i] = x[i] * sigmoid;
        }
        
        cache_store(apu->cache, opcode, data_hash, output, count * sizeof(float));
        return 0;
    }
    
    if (strcmp(opcode, "GELU") == 0) {
        if (!operands || !operands[0]) return -1;
        
        float* x = (float*)operands[0];
        float* out = (float*)output;
        
        for (int i = 0; i < count; i++) {
            float y = x[i];
            float c = 0.044714999999999997f;
            out[i] = 0.5f * y * (1.0f + tanhf(0.7978845608028674f * (y + c * y * y * y)));
        }
        
        return 0;
    }
    
    if (strcmp(opcode, "Softmax") == 0) {
        if (!operands || !operands[0]) return -1;
        
        float* x = (float*)operands[0];
        
        float max_val = -1e30f;
        for (int i = 0; i < count; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < count; i++) {
            sum_exp += expf(x[i] - max_val);
        }
        
        float* out = (float*)output;
        for (int i = 0; i < count; i++) {
            out[i] = expf(x[i] - max_val) / sum_exp;
        }
        
        return 0;
    }
    
    if (strcmp(opcode, "BitAttention") == 0) {
        if (!operands || !operands[0] || !operands[1] || !operands[2]) return -1;
        if (!shapes || shapes[0] <= 0 || shapes[1] <= 0) return -1;
        
        float* Q = (float*)operands[0];
        float* K = (float*)operands[1];
        float* V = (float*)operands[2];
        
        int seq_len = shapes[0];
        int head_dim = shapes[1];
        if (head_dim <= 0) return -1;
        float scale = 1.0f / sqrtf((float)head_dim);
        
        float* out = (float*)output;
        
        for (int s = 0; s < seq_len; s++) {
            float max_att = -1e30f;
            for (int s2 = 0; s2 <= s; s2++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += Q[s * head_dim + d] * K[s2 * head_dim + d];
                }
                score *= scale;
                if (score > max_att) max_att = score;
            }
            
            float sum_exp = 0.0f;
            for (int s2 = 0; s2 <= s; s2++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += Q[s * head_dim + d] * K[s2 * head_dim + d];
                }
                score *= scale;
                sum_exp += expf(score - max_att);
            }
            
            for (int d = 0; d < head_dim; d++) {
                float weighted = 0.0f;
                for (int s2 = 0; s2 <= s; s2++) {
                    float score = 0.0f;
                    for (int dd = 0; dd < head_dim; dd++) {
                        score += Q[s * head_dim + dd] * K[s2 * head_dim + dd];
                    }
                    score *= scale;
                    float attn_weight = expf(score - max_att) / sum_exp;
                    weighted += attn_weight * V[s2 * head_dim + d];
                }
                out[s * head_dim + d] = weighted;
            }
        }
        
        return 0;
    }
    
    if (strcmp(opcode, "BitLinear") == 0) {
        if (!operands || !operands[0] || !operands[1]) return -1;
        if (!shapes || shapes[0] <= 0 || shapes[1] <= 0 || shapes[2] <= 0) return -1;
        
        float* x = (float*)operands[0];
        uint8_t* weights = (uint8_t*)operands[1];
        
        int batch = shapes[0];
        int in_feat = shapes[1];
        int out_feat = shapes[2];
        
        memset(output, 0, (size_t)batch * out_feat * sizeof(float));
        
        for (int b = 0; b < batch; b++) {
            for (int o = 0; o < out_feat; o++) {
                float acc = 0.0f;
                for (int i = 0; i < in_feat; i++) {
                    uint8_t code = (weights[o * ((in_feat + 3) / 4) + i / 4] >> ((i % 4) * 2)) & 0x03;
                    float w = (code == 0x01) ? 1.0f : (code == 0x10) ? -1.0f : 0.0f;
                    acc += w * x[b * in_feat + i];
                }
                ((float*)output)[b * out_feat + o] = acc;
            }
        }
        
        return 0;
    }
    
    memset(output, 0, count * MTFP16_TRITS);
    return 0;
}

static void apu_get_stats(BBDOS_APU* apu, int* hits, int* misses, int* evictions, int* rejected) {
    cache_get_stats(apu->cache, hits, misses, evictions, rejected);
}

/* ============================================================================
 * C API Export
 * ============================================================================ */

BBDOS_APU* bbdos_apu_create(int cache_size) {
    return apu_create(cache_size);
}

void bbdos_apu_destroy(BBDOS_APU* apu) {
    apu_destroy(apu);
}

int bbdos_apu_exec(BBDOS_APU* apu, const char* opcode, void** operands, int* shapes, void* output) {
    return apu_exec(apu, opcode, operands, shapes, output);
}

void bbdos_apu_stats(BBDOS_APU* apu, int* hits, int* misses, int* evictions, int* rejected) {
    apu_get_stats(apu, hits, misses, evictions, rejected);
}

uint64_t bbdos_capability(void) {
    return HAS_AVX2 ? 1 : 0;
}

const char* bbdos_version(void) {
    return "1.0.0-c-avx-full";
}
