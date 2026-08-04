#include "data/Dataset.hpp"

#include "types/eigen_types.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

#include <Eigen/Dense>

Dataset::Dataset(Matrix&& dataset, IntVector&& target_data)
	: dataset_{std::move(dataset)},
	  target_data_{std::move(target_data)},
	  indices_(dataset_.cols()) {
	assert(dataset_.rows() > 0);
	assert(dataset_.cols() > 0);
	assert(indices_.size() == static_cast<size_t>(dataset_.cols()));
	std::iota(indices_.begin(), indices_.end(), 0);
	RandDataset();
}

Dataset::ConstBatchIterator Dataset::cbegin() const noexcept {
	return Dataset::ConstBatchIterator{0, this};
}

Dataset::ConstBatchIterator Dataset::cend() const noexcept {
	assert(indices_.size() < std::numeric_limits<int64_t>::max());
	return Dataset::ConstBatchIterator{static_cast<int64_t>(indices_.size()),
									   this};
}

Dataset::ConstBatchIterator Dataset::begin() const noexcept {
	return cbegin();
}

Dataset::ConstBatchIterator Dataset::end() const noexcept {
	return cend();
}

void Dataset::RandDataset() {
	assert(!indices_.empty());
	assert(indices_.size() == static_cast<size_t>(dataset_.cols()));
	std::random_device random{};
	std::mt19937 generator{random()};
	std::shuffle(indices_.begin(), indices_.end(), generator);
}

void Dataset::SetBatchSize(int64_t batch_size) noexcept {
	batch_size_ = batch_size;
}

int64_t Dataset::GetBatchSize() const noexcept {
	return batch_size_;
}

int64_t Dataset::GetSizeCols() const noexcept {
	return dataset_.cols();
}

int64_t Dataset::GetSizeRows() const noexcept {
	return dataset_.rows();
}
