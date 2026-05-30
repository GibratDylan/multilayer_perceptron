#include "trainer/metrics.hpp"

#include <Eigen/Dense>

namespace Metrics {
float accuracy(const Eigen::MatrixXd& predictive_inputs,
			   const Eigen::VectorXi& target_inputs) {
	assert(predictive_inputs.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= predictive_inputs.rows());

	assert(predictive_inputs.size() > 0 && target_inputs.size() > 0);

	Eigen::VectorXi predictions(target_inputs.size());
	for (int index = 0; index < target_inputs.size(); ++index) {
		predictive_inputs.col(index).maxCoeff(&predictions(index));
	}

	assert(predictions.size() > 0);

	return (predictions.array() == target_inputs.array()).cast<float>().mean();
}
}  // namespace Metrics
