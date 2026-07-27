#include "loss/LossCategoricalCrossEntropy.hpp"

#include "types/eigen_types.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>

void LossCategoricalCrossEntropy::Forward(const MatrixIn& logits_batch,
										  IntVectorIn targets_batch) {
	assert(logits_batch.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= logits_batch.rows());
	assert(logits_batch.size() > 0 && targets_batch.size() > 0);

	Inputs() = logits_batch;

	Vector correct_confidences(targets_batch.size());
	for (int64_t batch_index{}; batch_index < targets_batch.size();
		 ++batch_index) {
		correct_confidences(batch_index) =
			logits_batch(targets_batch(batch_index), batch_index);
	}

	assert(correct_confidences.size() > 0);
	constexpr float kClampEpsilon{1e-7F};

	for (auto& correct_confidence : correct_confidences)
		correct_confidence = std::max(
			kClampEpsilon, std::min(1.F - kClampEpsilon, correct_confidence));

	Outputs() = -(correct_confidences.array().log());

	assert(Outputs().rows() == targets_batch.rows());
}

void LossCategoricalCrossEntropy::Backward(IntVectorIn targets_batch) {
	assert(Inputs().cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= Inputs().rows());
	assert(Inputs().size() > 0 && targets_batch.size() > 0);

	InputsGradient().resizeLike(Inputs());
	InputsGradient().setZero();
	for (int64_t i{}; i < Inputs().cols(); ++i) {
		int64_t label{targets_batch(i)};
		float predictive_input{std::max(
			kClampEpsilon, std::min(1.F - kClampEpsilon, Inputs()(label, i)))};
		InputsGradient()(label, i) = -1.F / predictive_input;
	}
	InputsGradient() /= static_cast<float>(Inputs().cols());

	assert(InputsGradient().rows() == Inputs().rows() &&
		   InputsGradient().cols() == Inputs().cols());
}
