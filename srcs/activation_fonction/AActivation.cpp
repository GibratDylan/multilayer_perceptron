#include "../../includes/activation_fonction/AActivation.hpp"

#include <Eigen/Dense>

const Eigen::MatrixXd& AActivation::getOutputs() const {
	return _outputs;
}
