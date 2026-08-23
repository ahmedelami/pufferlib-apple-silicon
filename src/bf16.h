// Usage:
//     #include "bf16.h"
//
//     Set env_name/binding.c obs to PrecisionTensor
//
//     bf16* observations;
//     observations[0] = f32_to_bf16(some_float);                // scalar
//
//     // SIMD fast-path for inner loops with 8 floats already in an __m256:
//     __m256 v = _mm256_mul_ps(x, scale);
//     store_f32x8_as_bf16(&observations[i], v);                 // 1 store, 8 vals
//
//     // Reverse if you ever need to read back as float:
//     float f = bf16_to_f32(observations[0]);
//
// The vector helper uses AVX2 on x86_64 and two NEON vectors on AArch64.

#include <stdint.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

typedef uint16_t bf16;

static inline bf16 f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static inline float bf16_to_f32(bf16 b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

#if defined(__x86_64__) || defined(_M_X64)
static inline void store_f32x8_as_bf16(bf16* dst, __m256 v) {
    __m256i vi = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
    __m128i lo = _mm256_castsi256_si128(vi);
    __m128i hi = _mm256_extracti128_si256(vi, 1);
    _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(lo, hi));
}
#elif defined(__aarch64__) || defined(_M_ARM64)
static inline void store_f32x8_as_bf16(bf16* dst, float32x4x2_t v) {
    uint16x4_t lo = vshrn_n_u32(vreinterpretq_u32_f32(v.val[0]), 16);
    uint16x4_t hi = vshrn_n_u32(vreinterpretq_u32_f32(v.val[1]), 16);
    vst1q_u16(dst, vcombine_u16(lo, hi));
}
#else
static inline void store_f32x8_as_bf16(bf16* dst, const float v[8]) {
    for (int i = 0; i < 8; i++) dst[i] = f32_to_bf16(v[i]);
}
#endif
