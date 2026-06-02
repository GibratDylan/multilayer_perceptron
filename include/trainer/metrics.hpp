#pragma once

#include "types/eigen_types.hpp"

namespace metrics {
float Accuracy(const MatrixIn& logits_batch, IntVectorIn targets_batch);
}  // namespace metrics
