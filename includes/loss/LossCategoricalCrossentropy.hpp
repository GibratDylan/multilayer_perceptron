#pragma once
#include "ALoss.hpp"

#include <Eigen/Dense>

class LossCategoricalCrossentropy : public ALoss {
   private:
	void forward(const Eigen::MatrixXd& predictive_inputs,
				 const Eigen::VectorXi& target_inputs);
};
