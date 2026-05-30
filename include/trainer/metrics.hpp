#pragma once

#include "types/EigenTypes.hpp"
#include "types/Types.hpp"

namespace metrics {
using LogitsBatch = Matrix;
using TargetsBatch = IntVector;

using Logits = LogitsBatch;
using Targets = TargetsBatch;

AccuracyScore Accuracy(const LogitsBatch& logits_batch,
					   const TargetsBatch& targets_batch);
}  // namespace metrics
