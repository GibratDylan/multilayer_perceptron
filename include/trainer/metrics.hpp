#pragma once

#include "types/eigen_types.hpp"

namespace metrics {
float Accuracy(MatrixIn logits_batch, IntVectorIn targets_batch);
}  // namespace metrics
