#include "loss/LossCategoricalCrossEntropy.hpp"

void LossCategoricalCrossEntropy::Forward(const LogitsBatch& logits_batch,
										  const TargetsBatch& targets_batch) {
	assert(logits_batch.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= logits_batch.rows());
	assert(logits_batch.size() > 0 && targets_batch.size() > 0);

	inputs_ = logits_batch;

	Vector correct_confidences(targets_batch.size());
	for (Index batch_index = 0; batch_index < targets_batch.size();
		 ++batch_index) {
		correct_confidences(batch_index) =
			logits_batch(targets_batch(batch_index), batch_index);
	}

	assert(correct_confidences.size() > 0);

	for (auto& correct_confidence : correct_confidences)
		correct_confidence =
			std::max(1e-7, std::min(1 - 1e-7, correct_confidence));

	outputs_ = -(correct_confidences.array().log());

	assert(outputs_.rows() == targets_batch.rows());
}

void LossCategoricalCrossEntropy::Backward(const TargetsBatch& targets_batch) {
	assert(inputs_.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= inputs_.rows());
	assert(inputs_.size() > 0 && targets_batch.size() > 0);

	inputs_gradient_.resizeLike(inputs_);
	for (Index i = 0; i < inputs_.cols(); ++i) {
		Index label = static_cast<Index>(targets_batch(i));
		LossValue predictive_input =
			std::max(1e-7, std::min(1 - 1e-7, inputs_(label, i)));
		inputs_gradient_(label, i) = -1.0 / predictive_input;
	}
	inputs_gradient_ /= static_cast<LossValue>(inputs_.cols());

	assert(inputs_gradient_.rows() == inputs_.rows() &&
		   inputs_gradient_.cols() == inputs_.cols());
}
