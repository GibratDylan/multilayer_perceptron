#include "activation/ActivationReLU.hpp"

#include <Eigen/Dense>

void ActivationReLU::forward(const Eigen::MatrixXd& inputs) {
	inputs_ = inputs;
	outputs_ = inputs.cwiseMax(0);

	assert(inputs.rows() == outputs_.rows() &&
		   inputs.cols() == outputs_.cols());
	assert(outputs_.minCoeff() >= 0);
}

void ActivationReLU::backward(const Eigen::MatrixXd& inputs) {
	inputs_gradient_ = inputs;

	for (size_t i = 0, size = inputs_gradient_.size(); i < size; i++) {
		double temporary = (*(inputs_.data() + i));

		if (temporary < 0) *(inputs_gradient_.data() + i) = 0;
	}

	assert(inputs_gradient_.rows() == inputs.rows() &&
		   inputs_gradient_.cols() == inputs.cols());
}
