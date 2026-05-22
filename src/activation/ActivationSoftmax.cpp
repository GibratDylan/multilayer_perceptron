#include "activation/ActivationSoftmax.hpp"

#include "Eigen/src/Core/Matrix.h"

#include <Eigen/Dense>

void ActivationSoftmax::forward(const Eigen::MatrixXd& inputs) {
	inputs_ = inputs;
	outputs_ = inputs.rowwise() - inputs.colwise().maxCoeff();
	outputs_ = outputs_.array().exp();
	outputs_.array().rowwise() /= outputs_.colwise().sum().array();

	assert(inputs.rows() == outputs_.rows() &&
		   inputs.cols() == outputs_.cols());
}

// Doit etre tester !!!
void ActivationSoftmax::backward(const Eigen::MatrixXd& inputs) {
	inputs_gradient_.resizeLike(inputs);

	for (int i = 0; i < inputs.cols(); ++i) {
		Eigen::VectorXd s = outputs_.col(i);

		Eigen::MatrixXd jacobian =
			Eigen::MatrixXd(s.asDiagonal()) - s * s.transpose();

		inputs_gradient_.col(i) = jacobian * inputs.col(i);
	}

	assert(inputs_gradient_.rows() == inputs.rows() &&
		   inputs_gradient_.cols() == inputs.cols());
}
