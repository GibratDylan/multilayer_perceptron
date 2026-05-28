#include "NeuronalLayer.hpp"
#include "activation/ActivationReLU.hpp"
#include "activation/ActivationSoftmax.hpp"
#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"
#include "config/Config.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "network/Network.hpp"
#include "trainer/Metrics.hpp"
#include "trainer/observer/TrainerObserverMetricsWriter.hpp"

#include <iostream>
#include <string>

#include <Eigen/Dense>

int main(const int argc, const char** argv) {
	if (argc != 2) {
		std::cerr << "Program need config file: ./mlp path/to/config.txt\n";
		return 1;
	}

	Config config{std::string{argv[1]}};
	if (!config.parse()) return 1;

	/// inputs = rows(input), batch = cols(batch)
	Eigen::MatrixXd inputs(config.getNeuralLayer().front(),
						   config.getBatchSize());
	for (unsigned int i = 0; i < inputs.rows(); ++i) {
		for (unsigned int j = 0; j < inputs.cols(); ++j) inputs(i, j) = 0.5;
	}

	/// targets = rows(batch)
	Eigen::VectorXi targets(config.getBatchSize());
	for (unsigned int i = 0; i < targets.rows(); ++i) targets(i) = 1;

	Network network{std::make_unique<LossCategoricalCrossEntropy>()};

	for (unsigned int i = 0; i < config.getSize() - 2; ++i)
		network.addLayer(NeuronalLayer{config.getNeuralLayer()[i],
									   config.getNeuralLayer()[i + 1]},
						 std::make_unique<ActivationReLU>());

	network.addLayer(NeuronalLayer{64, 2},
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

// Eigen::MatrixXf or Eigen::MatrixXd ?
// Optimise memory usage
