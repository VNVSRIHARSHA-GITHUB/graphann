#pragma once

#include <cstdint>

// Squared L2 (Euclidean) distance between two float vectors.
// No sqrt — monotonic, so rankings are preserved.
// Compiler auto-vectorizes this with -O3 -march=native.
float compute_l2sq(const float* a, const float* b, uint32_t dim);
