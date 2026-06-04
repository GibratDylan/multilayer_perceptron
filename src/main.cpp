#include "config/Config.hpp"
#include "network/Network.hpp"
#include "network/NetworkBuilder.hpp"
#include "types/eigen_types.hpp"

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
	for (int64_t i{}; i < inputs.rows(); ++i) {
		for (int64_t j{}; j < inputs.cols(); ++j) {
			inputs(i, j) = 0.5F;
		}
	}

	/// targets = rows(batch)
	IntVector targets(config.GetBatchSize());
	for (int64_t i{}; i < targets.rows(); ++i) targets(i) = 1;

	NetworkBuilder network_builder{config};

	Network network{network_builder.Build()};

	std::cout << network.ForwardPass(inputs, targets) << '\n';
	network.BackwardPass(targets);
	std::cout << network.GetNeuronalLayers().front() << '\n';

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

// Chercher erreur optimisation
// Optimise memory usage (Ne plus stocker dans les backwardpass ? )
// BackwardPass
// Optimizer
// dataset_csv
// Rajouter et verifier les assert
// Catch exception main et exception safety program
// Release mode
