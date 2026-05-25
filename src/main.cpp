#include "NeuronalLayer.hpp"
#include "activation/ActivationReLU.hpp"
#include "activation/ActivationSoftmax.hpp"
#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "network/Network.hpp"
#include "trainer/Metrics.hpp"
#include "trainer/observer/TrainerObserverMetricsWriter.hpp"

#include <iostream>

#include <Eigen/Dense>

int main() {
	/// inputs = rows(input), batch = cols(batch)
	Eigen::MatrixXd inputs(2, 5);
	inputs << 1., 2., -1.5, 1., 2., 4., -2, .56, .23, 1.46;

	/// targets = rows(batch)
	Eigen::VectorXi targets(5);
	targets << 1, 0, 1, 1, 1;

	Network network{std::make_unique<LossCategoricalCrossEntropy>()};

	network.addLayer(NeuronalLayer{2, 10}, std::make_unique<ActivationReLU>());
	network.addLayer(NeuronalLayer{10, 2},
					 std::make_unique<ActivationSoftmax>());

	std::cout << network.forwardPass(inputs, targets) << '\n';

	// Layer dense_1{2, 10};
	// ActivationReLU activation_1{};
	// Layer dense_2{10, 2};
	// ActivationSoftmax activation_2{};
	// LossCategoricalCrossEntropy loss{};

	// std::cout << "\n\nForward:\n";
	// dense_1.forward(inputs);
	// activation_1.forward(dense_1.getOutputs());
	// dense_2.forward(activation_1.getOutputs());
	// std::cout << dense_2 << '\n';
	// activation_2.forward(dense_2.getOutputs());
	// std::cout << activation_2 << '\n';
	// loss.forward(activation_2.getOutputs(), targets);
	// std::cout << loss << '\n';

	// std::cout << "\n\nBackward:\n";
	// loss.backward(targets);
	// std::cout << loss << '\n';
	// activation_2.backward(loss.getInputsGradient());
	// std::cout << activation_2 << '\n';
	// dense_2.backward(activation_2.getInputsGradient());
	// std::cout << dense_2 << '\n';

	return 0;
}

// Use Eigen::matrix or Eigen::vector everywhere ?
// _outputs in class ?
// Install and download libs with make
// Check Matrix and vector type
