#pragma once

#include "types/eigen_types.hpp"

#include <random>

#include <Eigen/Dense>

class Dataset {
   public:
	class BatchIterator {
	   private:
		int64_t pos_;
		int64_t batch_size_;

	   public:
		explicit BatchIterator(int64_t pos) : pos_{pos} {
			// std::iota(indices_.begin(), indices_.end(), 0);
			// std::random_device random;
			// std::mt19937 generator(random());

			// std::shuffle(indices_.cbegin(), indices_.cend(), generator);
		}

		MatrixIn operator*() const {}

		const BatchIterator& operator++() {
			pos_ += batch_size_;
			return *this;
		}

		bool operator!=(const BatchIterator& rhs) const {
			return rhs.pos_ != pos_;
		}
	};

   private:
	const Matrix dataset_;
	std::vector<int64_t> indices_;
	int64_t batch_size_;

   public:
	explicit Dataset(const Matrix&& dataset);

	const BatchIterator cbegin() const;
	const BatchIterator cend() const;

	void RandDataset();
};
