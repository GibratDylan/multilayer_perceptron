#include "Layer.hpp"
#include "activation/ActivationReLU.hpp"
#include "activation/ActivationSoftmax.hpp"
#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "trainer/Metrics.hpp"
#include "trainer/observer/TrainerObserverMetricsWriter.hpp"

#include <iostream>

#include <Eigen/Dense>

int main() {
	/// inputs = rows(input), batch = cols(batch)
	Eigen::MatrixXd inputs(2, 5);
	inputs << 1., 2., -1.5, 1., 2., 4., -2, .56, .23, 1.46;

	Layer dense_1{2, 10};
	ActivationReLU activation_1{};
	Layer dense_2{10, 2};
	// ActivationSoftmax activation_2{};
	dense_1.forward(inputs);
	activation_1.forward(dense_1.getOutputs());

	dense_2.forward(activation_1.getOutputs());
	std::cout << dense_2 << '\n';
	// activation_2.forward(dense_2.getOutputs());

	/// inputs = rows(input), batch = cols(batch)
	// Eigen::MatrixXd inputs(2, 5);
	// inputs << 1., 2., -1.5, 1., 2., 4., -2, .56, .23, 1.46;

	/// targets = rows(batch)
	Eigen::VectorXi targets(5);
	targets << 1, 0, 1, 1, 1;

	// std::cout << activation_2.getOutputs() << '\n';
	ActivationSoftmaxLossCategoricalCrossentropy softmax_loss{};
	// softmax_loss.calculate(inputs, targets);

	std::cout << "Loss mean: "
			  << softmax_loss.forward(dense_2.getOutputs(), targets) << '\n';
	std::cout << "Accuracy: "
			  << Metrics::accuracy(softmax_loss.getOutputs(), targets) << '\n';
	std::cout << softmax_loss << '\n';

	softmax_loss.backward(targets);
	std::cout << softmax_loss << '\n';

	dense_2.backward(softmax_loss.getInputsGradient());
	std::cout << dense_2 << '\n';

	// TrainerObserverMetricsWriter metrics_writer;
	// metrics_writer.on_epoch_end(0,
	// Metrics::accuracy(activation2.getOutputs(), targets));

	return 0;
}

// Use Eigen::matrix or Eigen::vector everywhere ?
// _outputs in class ?
// Install and download libs with make
// Check Matrix and vector type
