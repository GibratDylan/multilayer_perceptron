#include "activation/ActivationSoftmax.hpp"

void ActivationSoftmax::Forward(MatrixIn input_batch) {
	inputs_ = input_batch;
	outputs_ = input_batch.rowwise() - input_batch.colwise().maxCoeff();
	outputs_ = outputs_.array().exp();
	outputs_.array().rowwise() /= outputs_.colwise().sum().array();

	assert(input_batch.rows() == outputs_.rows() &&
		   input_batch.cols() == outputs_.cols());
}

void ActivationSoftmax::Backward(MatrixIn gradient_batch) {
	inputs_gradient_.resizeLike(gradient_batch);
	for (int64_t i = 0; i < gradient_batch.cols(); ++i) {
		Vector s = outputs_.col(i);

		Matrix jacobian = Matrix(s.asDiagonal());
		jacobian.noalias() -= s * s.transpose();

		inputs_gradient_.col(i).noalias() = jacobian * gradient_batch.col(i);
	}

	assert(inputs_gradient_.rows() == gradient_batch.rows() &&
		   inputs_gradient_.cols() == gradient_batch.cols());
}
