#include "../../includes/activation/ActivationReLU.hpp"

#include <Eigen/Dense>

void ActivationReLU::forward(const Eigen::MatrixXd& inputs) {
	_outputs = inputs.cwiseMax(0);
	assert(inputs.rows() == _outputs.rows() &&
		   inputs.cols() == _outputs.cols());
	assert(_outputs.minCoeff() >= 0);
}
