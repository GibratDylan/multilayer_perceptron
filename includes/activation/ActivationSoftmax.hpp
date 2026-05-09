#pragma once
#include "AActivation.hpp"

#include <Eigen/Dense>

class ActivationSoftmax : public AActivation {
   public:
	void forward(const Eigen::MatrixXd& inputs);
};
