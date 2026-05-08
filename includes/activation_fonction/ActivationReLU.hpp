#pragma once
#include "AActivation.hpp"

class ActivationReLU : public AActivation {
   public:
	void forward(const Eigen::MatrixXd& inputs);
};
