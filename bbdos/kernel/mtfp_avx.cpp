#include "mtfp_avx.h"
#include <string.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

static inline int8_t float_to_trit(float value, int n_trits) {
    if (value >= 1.0f) return 1;
    if (value <= -1.0f) return -1;
    return 0;
}

static inline float unpack_single_mtfp(
    const int8_t* trits,
    int n_sign,
    int n_exponent,
    int n_mantissa
) {
    if (trits[n_sign] == 1 && 
        trits[n_sign + 1] == 2 && 
        trits[n_sign + 2] == 2) {
        return INFINITY;
    }
    
    int sign = trits[0] == 1 ? -1 : 1;
    
    int exp = 0;
    for (int i = 0; i < n_exponent; i++) {
        int t = trits[n_sign + i];
        if (t < 0) t += 3;
        exp += t * (int)pow(3, i);
    }
    
    int mantissa = 0;
    for (int i = 0; i < n_mantissa; i++) {
        int t = trits[n_sign + n_exponent + i];
        if (t < 0) t += 3;
        mantissa += t * (int)pow(3, n_mantissa - 1 - i);
    }
    
    float result = (float)mantissa / pow(3, n_mantissa - 1);
    result *= pow(3.0f, exp);
    
    return sign * result;
}

static inline void pack_single_mtfp(
    float value,
    int8_t* trits,
    int n_sign,
    int n_exponent,
    int n_mantissa
) {
    memset(trits, 0, n_sign + n_exponent + n_mantissa);
    
    if (value == 0.0f) return;
    
    int sign = 0;
    if (value < 0) {
        sign = 1;
        value = -value;
    }
    
    int exp = 0;
    if (value >= 1.0f) {
        while (value >= 3.0f && exp < 80) {
            value /= 3.0f;
            exp++;
        }
    } else {
        while (value < 1.0f && exp > -80) {
            value *= 3.0f;
            exp--;
        }
    }
    
    int mantissa = (int)(value * pow(3, n_mantissa - 1));
    
    trits[0] = sign;
    
    for (int i = 0; i < n_exponent; i++) {
        trits[n_sign + i] = exp % 3;
        exp /= 3;
    }
    
    for (int i = 0; i < n_mantissa; i++) {
        trits[n_sign + n_exponent + i] = mantissa % 3;
        mantissa /= 3;
    }
}

#if USE_AVX
extern "C" void mtfp_pack_avx(
    const float* input,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    
    for (int i = 0; i < count; i++) {
        pack_single_mtfp(
            input[i],
            &output[i * n_trits],
            1,
            n_exponent_trits,
            n_mantissa_trits
        );
    }
}

extern "C" void mtfp_unpack_avx(
    const int8_t* input,
    float* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    
    for (int i = 0; i < count; i++) {
        output[i] = unpack_single_mtfp(
            &input[i * n_trits],
            1,
            n_exponent_trits,
            n_mantissa_trits
        );
    }
}

extern "C" void mtfp_add_avx(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    
    for (int i = 0; i < count; i++) {
        float av = unpack_single_mtfp(&a[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        float bv = unpack_single_mtfp(&b[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        pack_single_mtfp(av + bv, &output[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
    }
}

extern "C" void mtfp_mul_avx(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    
    for (int i = 0; i < count; i++) {
        float av = unpack_single_mtfp(&a[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        float bv = unpack_single_mtfp(&b[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        pack_single_mtfp(av * bv, &output[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
    }
}

#else

extern "C" void mtfp_pack_avx(
    const float* input,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    
    for (int i = 0; i < count; i++) {
        pack_single_mtfp(
            input[i],
            &output[i * n_trits],
            1,
            n_exponent_trits,
            n_mantissa_trits
        );
    }
}

extern "C" void mtfp_unpack_avx(
    const int8_t* input,
    float* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    for (int i = 0; i < count; i++) {
        output[i] = unpack_single_mtfp(&input[i * (1 + n_exponent_trits + n_mantissa_trits)], 1, n_exponent_trits, n_mantissa_trits);
    }
}

extern "C" void mtfp_add_avx(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    for (int i = 0; i < count; i++) {
        float av = unpack_single_mtfp(&a[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        float bv = unpack_single_mtfp(&b[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        pack_single_mtfp(av + bv, &output[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
    }
}

extern "C" void mtfp_mul_avx(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
) {
    int n_trits = 1 + n_exponent_trits + n_mantissa_trits;
    for (int i = 0; i < count; i++) {
        float av = unpack_single_mtfp(&a[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        float bv = unpack_single_mtfp(&b[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
        pack_single_mtfp(av * bv, &output[i * n_trits], 1, n_exponent_trits, n_mantissa_trits);
    }
}

#endif

extern "C" uint64_t mtfp_get_capability(void) {
#if USE_AVX
    return 0x00000001;
#else
    return 0x00000000;
#endif
}
