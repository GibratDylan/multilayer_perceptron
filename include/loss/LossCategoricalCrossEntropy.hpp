#pragma once

#include "ALoss.hpp"

#include <Eigen/Dense>

class LossCategoricalCrossEntropy : public ALoss {
   public:
	void forward(const Eigen::MatrixXd& predictive_inputs,
				 const Eigen::VectorXi& target_inputs) override;
	void backward(const Eigen::VectorXi& target_inputs) override;
};
