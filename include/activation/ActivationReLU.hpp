#pragma once

#include "AActivation.hpp"

#include <Eigen/Dense>

class ActivationReLU : public AActivation {
   public:
	void forward(const Eigen::MatrixXd& inputs) override;
};
