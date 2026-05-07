#pragma once

#include <Eigen/Dense>
#include <ostream>

class Layer {
   private:
	Eigen::MatrixXd _weights;
	Eigen::VectorXd _biases;
	Eigen::VectorXd _outputs;

   public:
	Layer(int nb_inputs, int nb_neurons);
	void forward(const Eigen::VectorXd& inputs);
	friend std::ostream& operator<<(std::ostream& os, const Layer& rhs);
};
