#pragma once

#include "activation/ActivationSoftmax.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"

#include <ostream>

#include <Eigen/Dense>

class ActivationSoftmaxLossCategoricalCrossentropy {
   private:
	LossCategoricalCrossEntropy loss_ = LossCategoricalCrossEntropy();
	ActivationSoftmax activation_ = ActivationSoftmax();
	Eigen::MatrixXd outputs_;

	Eigen::MatrixXd inputs_gradient_;

   public:
	double forward(const Eigen::MatrixXd& inputs,
				   const Eigen::VectorXi& target_inputs);
	void backward(const Eigen::VectorXi& target_inputs);

	const Eigen::MatrixXd& getOutputs() const;
	const Eigen::MatrixXd& getInputsGradient() const;

	friend std::ostream& operator<<(
		std::ostream& os,
		const ActivationSoftmaxLossCategoricalCrossentropy& rhs);
};
