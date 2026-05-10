#pragma once

#include "ALoss.hpp"

#include <Eigen/Dense>

class LossCategoricalCrossEntropy : public ALoss {
   private:
	void forward(const Eigen::MatrixXd& predictive_inputs,
				 const Eigen::VectorXi& target_inputs) override;
};
