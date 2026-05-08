#include "../includes/activation_fonction/ActivationSoftmax.hpp"

#include <iostream>

#include <Eigen/Dense>

int main() {
	Eigen::MatrixXd X(4, 3);
	X << 1., 2., -1.5, 1., 2., 4., -2, .56, .23, 1.46, -.67, .12;

	std::cout << X << "\n";

	ActivationSoftmax relu{};
	relu.forward(X);

	std::cout << relu.getOutputs() << "\n";

	// layer.forward(X);

	// std::cout << layer.getOutputs() << "\n";

	return 0;
}
