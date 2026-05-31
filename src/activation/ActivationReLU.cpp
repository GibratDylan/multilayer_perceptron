#include "activation/ActivationReLU.hpp"

void ActivationReLU::Forward(MatrixIn input_batch) {
	inputs_ = input_batch;
	outputs_ = input_batch.cwiseMax(0.F);

	assert(input_batch.rows() == outputs_.rows() &&
		   input_batch.cols() == outputs_.cols());
	assert(outputs_.minCoeff() >= 0.F);
}

void ActivationReLU::Backward(MatrixIn gradient_batch) {
	inputs_gradient_ = gradient_batch;
	for (int64_t i = 0, size = inputs_gradient_.size(); i < size; ++i) {
		Matrix::Scalar temporary = (*(inputs_.data() + i));

		if (temporary < 0) *(inputs_gradient_.data() + i) = 0.F;
	}

	assert(inputs_gradient_.rows() == gradient_batch.rows() &&
		   inputs_gradient_.cols() == gradient_batch.cols());
}
