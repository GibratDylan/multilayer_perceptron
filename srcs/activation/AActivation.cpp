#include "../../includes/activation/AActivation.hpp"

#include <Eigen/Dense>

const Eigen::MatrixXd& AActivation::getOutputs() const {
	return _outputs;
}
