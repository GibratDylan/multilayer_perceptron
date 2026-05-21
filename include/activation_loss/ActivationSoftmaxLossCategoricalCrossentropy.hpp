#pragma once

#include "activation/ActivationSoftmax.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"

#include <Eigen/Dense>

class ActivationSoftmaxLossCategoricalCrossentropy {
   private:
	LossCategoricalCrossEntropy _loss = LossCategoricalCrossEntropy();
	ActivationSoftmax _activation = ActivationSoftmax();
	Eigen::MatrixXd _outputs;
	Eigen::MatrixXd _inputGradient;

   public:
	double forward(const Eigen::MatrixXd& inputs,
				   const Eigen::VectorXi& target_inputs);

   void backward(const Eigen::VectorXi& target_inputs);

	const Eigen::MatrixXd& getOutputs() const;

	const Eigen::MatrixXd& getInputsGradient() const;
};
