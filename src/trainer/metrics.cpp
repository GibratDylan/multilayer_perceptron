#include "trainer/metrics.hpp"

namespace metrics {
float Accuracy(const MatrixIn& logits_batch, IntVectorIn targets_batch) {
	assert(logits_batch.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= logits_batch.rows());

	assert(logits_batch.size() > 0 && targets_batch.size() > 0);

	IntVector predictions(targets_batch.size());
	for (int64_t index{}; index < targets_batch.size(); ++index) {
		logits_batch.col(index).maxCoeff(&predictions(index));
	}

	assert(predictions.size() > 0);

	return (predictions.array() == targets_batch.array()).cast<float>().mean();
}
}  // namespace metrics
