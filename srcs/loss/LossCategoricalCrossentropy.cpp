#include "../../includes/loss/LossCategoricalCrossentropy.hpp"

#include <Eigen/Dense>

void LossCategoricalCrossentropy::forward(
	const Eigen::MatrixXd& predictive_inputs,
	const Eigen::VectorXi& target_inputs) {
	assert(predictive_inputs.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= predictive_inputs.rows());

	Eigen::VectorXd correct_confidences(target_inputs.size());
	for (int index = 0; index < target_inputs.size(); ++index) {
		correct_confidences(index) =
			predictive_inputs(target_inputs(index), index);
	}

	for (auto& correct_confidence : correct_confidences) {
		correct_confidence = std::max(1e-7, correct_confidence);
		correct_confidence = std::min(1 - 1e-7, correct_confidence);
	}

	_outputs = -(correct_confidences.array().log());
	assert(_outputs.rows() == target_inputs.rows());
}
