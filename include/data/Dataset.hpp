#pragma once

#include "types/eigen_types.hpp"

#include <limits>

#include <Eigen/Dense>

class Dataset {
   public:
	class ConstBatchIterator {
	   private:
		int64_t pos_{};
		const Dataset* parent_ptr_{};

	   public:
		explicit ConstBatchIterator(int64_t pos,
									const Dataset* parent_ptr) noexcept
			: pos_{pos}, parent_ptr_{parent_ptr} {}

		Matrix operator*() const {
			assert(pos_ >= 0 &&
				   pos_ < static_cast<int64_t>(parent_ptr_->indices_.size()) &&
				   parent_ptr_->indices_.size() <
					   std::numeric_limits<int64_t>::max());

			const int64_t end_pos{
				std::min(pos_ + parent_ptr_->batch_size_,
						 static_cast<int64_t>(parent_ptr_->indices_.size()))};

			Matrix batch(parent_ptr_->dataset_.rows(), end_pos - pos_);

			for (int64_t index{0}, start_pos{pos_}; start_pos < end_pos;
				 ++start_pos, ++index)
				batch.col(index) =
					parent_ptr_->dataset_.col(parent_ptr_->indices_[start_pos]);

			return batch;
		}

		const ConstBatchIterator& operator++() noexcept {
			pos_ = std::min(pos_ + parent_ptr_->batch_size_,
							static_cast<int64_t>(parent_ptr_->indices_.size()));
			return *this;
		}

		bool operator!=(const ConstBatchIterator& rhs) const noexcept {
			return rhs.pos_ != pos_;
		}
	};

   private:
	const Matrix dataset_{};
	std::vector<int64_t> indices_{};
	int64_t batch_size_{};

   public:
	explicit Dataset(Matrix&& dataset, int64_t batch_size);

	// NOLINTBEGIN(readability-identifier-naming*)
	ConstBatchIterator cbegin() const noexcept;
	ConstBatchIterator cend() const noexcept;
	ConstBatchIterator begin() const noexcept;
	ConstBatchIterator end() const noexcept;
	// NOLINTEND(readability-identifier-naming*)

	void RandDataset();
};
