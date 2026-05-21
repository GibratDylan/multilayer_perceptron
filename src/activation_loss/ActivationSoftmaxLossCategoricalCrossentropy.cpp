#include "activation_loss/ActivationSoftmaxLossCategoricalCrossentropy.hpp"

#include <Eigen/Dense>

double ActivationSoftmaxLossCategoricalCrossentropy::forward(
	const Eigen::MatrixXd& inputs, const Eigen::VectorXi& target_inputs) {
	_activation.forward(inputs);
	_outputs = _activation.getOutputs();

	return _loss.calculate(_outputs, target_inputs);
}

void ActivationSoftmaxLossCategoricalCrossentropy::backward(const Eigen::VectorXi& target_inputs) {
    _inputGradient = _outputs;

    const Eigen::Index samples = target_inputs.size();

	assert(_inputGradient.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= _inputGradient.rows());

	assert(_inputGradient.size() > 0 && target_inputs.size() > 0);

    for (Eigen::Index index = 0; index < samples; ++index) {
        _inputGradient(target_inputs(index), index) -= 1.0;
    }

	_inputGradient /= static_cast<double>(samples);
}

const Eigen::MatrixXd&
ActivationSoftmaxLossCategoricalCrossentropy::getOutputs() const {
	return _outputs;
}

const Eigen::MatrixXd&
ActivationSoftmaxLossCategoricalCrossentropy::getInputsGradient() const {
	return _inputGradient;
}

