#pragma once

#include "types/EigenTypes.hpp"

namespace metrics {
float Accuracy(MatrixIn logits_batch, IntVectorIn targets_batch);
}  // namespace metrics
