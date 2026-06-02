#include "data/Dataset.hpp"

#include "types/eigen_types.hpp"

#include <cassert>
#include <cstdint>
#include <random>

#include <Eigen/Dense>

Dataset::Dataset(const Matrix&& dataset, int64_t batch_size)
	: dataset_{dataset}, indices_(dataset.cols()), batch_size_{batch_size} {
	assert(batch_size_ > 0);
	assert(dataset_.rows() > 0);
	assert(dataset_.cols() > 0);
	assert(indices_.size() == static_cast<size_t>(dataset_.cols()));
	std::iota(indices_.begin(), indices_.end(), 0);
	RandDataset();
}

Dataset::ConstBatchIterator Dataset::cbegin() const {
	return Dataset::ConstBatchIterator{0, this};
}

Dataset::ConstBatchIterator Dataset::cend() const {
	assert(indices_.size() < std::numeric_limits<int64_t>::max());
	return Dataset::ConstBatchIterator{static_cast<int64_t>(indices_.size()),
									   this};
}

Dataset::ConstBatchIterator Dataset::begin() const {
	return cbegin();
}

Dataset::ConstBatchIterator Dataset::end() const {
	return cend();
}

void Dataset::RandDataset() {
	assert(!indices_.empty());
	assert(indices_.size() == static_cast<size_t>(dataset_.cols()));
	std::random_device random{};
	std::mt19937 generator{random()};
	std::shuffle(indices_.begin(), indices_.end(), generator);
}
