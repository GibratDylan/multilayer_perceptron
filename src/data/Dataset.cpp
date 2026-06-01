#pragma once

#include "data/Dataset.hpp"

#include "types/eigen_types.hpp"

#include <Eigen/Dense>

Dataset::Dataset(const Matrix&& dataset) {}

const Dataset::BatchIterator& Dataset::BatchBegin(int64_t batch_size) const {}
const Dataset::BatchIterator& Dataset::BatchEnd(int64_t batch_size) const {}

void Dataset::RandDataset(int64_t batch_size) {}
