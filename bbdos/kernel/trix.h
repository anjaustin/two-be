#ifndef TRIX_H
#define TRIX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void trix_forward(
    const float* input,          // [batch_size, in_features]
    const uint8_t* packed_w,     // [num_tiles, out_per_tile, packed_in_dim]
    const float* scales,         // [num_tiles, out_per_tile]
    const int8_t* gate_mask,     // [batch_size, num_tiles]
    float* output,               // [batch_size, out_features]
    int batch_size,
    int in_features,
    int out_features,
    int num_tiles
);

void pack_weights(
    const float* raw_weights,    // {-1, 0, 1} floats
    uint8_t* packed_output,
    int rows,
    int cols
);

void unpack_weights(
    const uint8_t* packed_weights,
    float* raw_output,
    int rows,
    int cols
);

#ifdef __cplusplus
}
#endif

#endif
