#pragma once
#include "AActivation.hpp"

class ActivationSoftmax : public AActivation {
   public:
	void forward(const Eigen::MatrixXd& inputs);
};
