#ifndef NIYAH_MATH_INTERNAL_H
#define NIYAH_MATH_INTERNAL_H

#include <stddef.h>

float niyah_dot_f32(const float *a, const float *b, size_t n);
void niyah_axpy_f32(float *dst, const float *src, float scale, size_t n);

#endif
