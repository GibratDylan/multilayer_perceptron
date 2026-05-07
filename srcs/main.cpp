#include <Eigen/Dense>

#include "../includes/Layer.hpp"

// #include <algorithm>
// #include <array>
// #include <execution>
// #include <functional>
#include <iostream>

// #include <numeric>

int main() {
	Eigen::VectorXd inputs(4);
	inputs << 1., 2., -1.5, 1.;

	// Eigen::MatrixXd weights1(4, 3);  // (n_neurons, n_features)
	// weights1 << 0.2,   0.8,  -0.5,  1.0,   // neurone 0
	// 		0.5,  -0.91,  0.26, -0.5,   // neurone 1
	// 		-0.26, -0.27,  0.17,  0.87;  // neurone 2

	// Eigen::VectorXd biases1(3);
	// biases1 << 2., 3., .5;

	// Eigen::MatrixXd weights2(3, 3);  // (n_neurons, n_features)
	// weights2 << .1,   -.14,  .5,   // neurone 0
	// 		-.5,  .12,  -.33,   // neurone 1
	// 		-.44, .73,  -.13;  // neurone 2

	// Eigen::VectorXd biases2(3);
	// biases2 << -1., 2., -.5;

	// Eigen::MatrixXd outputs1 = weights1 * inputs;
	// outputs1.colwise() += biases1;

	// Eigen::MatrixXd outputs2 = weights2 * outputs1;
	// outputs2.colwise() += biases2;

	Layer layer{4, 3};
	std::cout << layer << "\n";

	layer.forward(inputs);

	std::cout << layer << "\n";

	return 0;
}
