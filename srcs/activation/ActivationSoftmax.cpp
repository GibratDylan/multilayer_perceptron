#include "../../includes/activation/ActivationSoftmax.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

void ActivationSoftmax::forward(const Eigen::MatrixXd& inputs) {
	_outputs = inputs.rowwise() - inputs.colwise().maxCoeff();
	_outputs = _outputs.array().exp();
	_outputs.array().rowwise() /= _outputs.colwise().sum().array();
	assert(inputs.rows() == _outputs.rows() &&
		   inputs.cols() == _outputs.cols());
}
