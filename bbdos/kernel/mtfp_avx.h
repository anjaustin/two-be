#ifndef MTFP_AVX_H
#define MTFP_AVX_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void mtfp_pack_avx(
    const float* input,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
);

void mtfp_unpack_avx(
    const int8_t* input,
    float* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
);

void mtfp_add_avx(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
);

void mtfp_mul_avx(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    int count,
    int n_mantissa_trits,
    int n_exponent_trits
);

#ifdef __cplusplus
}
#endif

#endif
