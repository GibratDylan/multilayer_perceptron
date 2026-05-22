#pragma once

#include <ostream>

#include <Eigen/Dense>

class AActivation {
   protected:
	Eigen::MatrixXd outputs_;
	Eigen::MatrixXd inputs_;

	Eigen::MatrixXd inputs_gradient_;

   public:
	virtual void forward(const Eigen::MatrixXd& inputs) = 0;
	virtual void backward(const Eigen::MatrixXd& inputs) = 0;
	virtual ~AActivation() = default;

	const Eigen::MatrixXd& getOutputs() const;
	const Eigen::MatrixXd& getInputsGradient() const;
};

std::ostream& operator<<(std::ostream& os, const AActivation& rhs);
