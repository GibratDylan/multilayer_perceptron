#pragma once

#include <ostream>

#include <Eigen/Dense>

class Layer {
   private:
	Eigen::MatrixXd weights_;
	Eigen::VectorXd biases_;
	Eigen::MatrixXd outputs_;
	Eigen::MatrixXd inputs_;

	Eigen::MatrixXd weights_gradient_;
	Eigen::VectorXd biases_gradient_;
	Eigen::MatrixXd inputs_gradient_;

   public:
	Layer(int input_size, int num_neurons);

	void forward(const Eigen::MatrixXd& inputs);
	void backward(const Eigen::MatrixXd& inputs);
	const Eigen::MatrixXd& getOutputs() const;
	const Eigen::MatrixXd& getInputsGradient() const;

	friend std::ostream& operator<<(std::ostream& os, const Layer& rhs);
};
