#pragma once

#include <ostream>

#include <Eigen/Dense>

class Layer {
   private:
	Eigen::MatrixXd _weights;
	Eigen::VectorXd _biases;
	Eigen::MatrixXd _outputs;

   public:
	Layer(int size_inputs, int nb_neurons);

	void forward(const Eigen::MatrixXd& inputs);
	const Eigen::MatrixXd& getOutputs() const;

	friend std::ostream& operator<<(std::ostream& os, const Layer& rhs);
};
