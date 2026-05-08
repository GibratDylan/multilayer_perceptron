#include "../../includes/activation_fonction/ActivationSoftmax.hpp"

#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

void ActivationSoftmax::forward(const Eigen::MatrixXd& inputs) {
	_outputs = inputs.colwise() - inputs.rowwise().maxCoeff();
	_outputs.colwise() = Eigen::exp(_outputs.array());
	std::cout << _outputs << '\n';
	assert(inputs.rows() == _outputs.rows() &&
		   inputs.cols() == _outputs.cols());
}
