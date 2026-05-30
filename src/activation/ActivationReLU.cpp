#include "activation/ActivationReLU.hpp"

void ActivationReLU::Forward(const InputBatch& input_batch) {
	inputs_ = input_batch;
	outputs_ = input_batch.cwiseMax(0);

	assert(input_batch.rows() == outputs_.rows() &&
		   input_batch.cols() == outputs_.cols());
	assert(outputs_.minCoeff() >= 0);
}

void ActivationReLU::Backward(const GradientBatch& gradient_batch) {
	inputs_gradient_ = gradient_batch;
	for (Index i = 0, size = inputs_gradient_.size(); i < size; ++i) {
		InputBatch::Scalar temporary = (*(inputs_.data() + i));

		if (temporary < 0) *(inputs_gradient_.data() + i) = 0;
	}

	assert(inputs_gradient_.rows() == gradient_batch.rows() &&
		   inputs_gradient_.cols() == gradient_batch.cols());
}
