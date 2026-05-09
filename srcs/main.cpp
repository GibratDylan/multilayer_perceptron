#include "../includes/Layer.hpp"
#include "../includes/activation/ActivationReLU.hpp"
#include "../includes/activation/ActivationSoftmax.hpp"
#include "../includes/loss/LossCategoricalCrossentropy.hpp"
#include "Eigen/src/Core/Matrix.h"

#include <iostream>

#include <Eigen/Dense>

int main() {
	/// x = rows(input), y = cols(batch)
	Eigen::MatrixXd X(2, 5);
	X << 1., 2., -1.5, 1., 2., 4., -2, .56, .23, 1.46;

	Layer dense1{2, 1000000};
	ActivationReLU activation1{};
	Layer dense2{1000000, 2};
	ActivationSoftmax activation2{};
	dense1.forward(X);
	activation1.forward(dense1.getOutputs());
	dense2.forward(activation1.getOutputs());
	activation2.forward(dense2.getOutputs());

	/// x = rows(input), y = cols(batch)
	// Eigen::MatrixXd X(2, 5);
	// X << 1., 2., -1.5, 1., 2., 4., -2, .56, .23, 1.46;

	/// target x = rows(batch)
	Eigen::VectorXi y(5);
	y << 1, 0, 1, 1, 1;

	std::cout << activation2.getOutputs() << '\n';
	LossCategoricalCrossentropy cross{};
	// cross.calculate(X, y);

	std::cout << cross.calculate(activation2.getOutputs(), y) << '\n';

	return 0;
}

// Use Eigen::matrix or Eigen::vector everywhere ?
// _outputs in class ?
// Install and download libs with make
// Check Matrix and vector type
