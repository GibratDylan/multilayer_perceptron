#include "NeuronalLayer.hpp"
#include "activation/ActivationReLU.hpp"
#include "activation/ActivationSoftmax.hpp"
#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"
#include "config/Config.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "network/Network.hpp"
#include "trainer/metrics.hpp"
#include "trainer/observer/TrainerObserverMetricsWriter.hpp"
#include "types/EigenTypes.hpp"

#include <iostream>
#include <string>

int main(int argc, const char** argv) {
	if (argc != 2) {
		std::cerr << "Program need config file: ./mlp path/to/config.txt\n";
		return 1;
	}

	Config config{std::string{argv[1]}};
	if (!config.Parse()) return 1;

	/// inputs = rows(input), batch = cols(batch)
	Matrix inputs(config.GetNeuralLayer().front(), config.GetBatchSize());
	for (Index i = 0; i < inputs.rows(); ++i) {
		for (Index j = 0; j < inputs.cols(); ++j) inputs(i, j) = 0.5;
	}

	/// targets = rows(batch)
	IntVector targets(config.GetBatchSize());
	for (Index i = 0; i < targets.rows(); ++i) targets(i) = 1;

	Network network{std::make_unique<LossCategoricalCrossEntropy>()};

	for (LayerIndex i = 0; i < config.GetSize() - 2; ++i)
		network.AddLayer(NeuronalLayer{config.GetNeuralLayer()[i],
									   config.GetNeuralLayer()[i + 1]},
						 std::make_unique<ActivationReLU>());

	network.AddLayer(NeuronalLayer{64, 2},
					 std::make_unique<ActivationSoftmax>());

	std::cout << network.ForwardPass(inputs, targets) << '\n';

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
