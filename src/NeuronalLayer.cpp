#include "NeuronalLayer.hpp"

#include <ostream>

#include <Eigen/Dense>

NeuronalLayer::NeuronalLayer(size_t input_size, size_t num_neurons)
	: weights_{Eigen::MatrixXd::Random(static_cast<long>(num_neurons),
									   static_cast<long>(input_size))},
	  biases_(num_neurons) {
	weights_ *= 0.10;
	biases_.setZero();
}

void NeuronalLayer::forward(const Eigen::MatrixXd& inputs) {
	assert(inputs.rows() == weights_.cols() && inputs.size() > 0);

	inputs_ = inputs;
	outputs_ = weights_ * inputs;

	assert(outputs_.rows() == biases_.rows());

	outputs_.colwise() += biases_;

	assert(inputs.cols() == outputs_.cols() &&
		   weights_.rows() == outputs_.rows());
}

void NeuronalLayer::backward(const Eigen::MatrixXd& inputs) {
	weights_gradient_ = inputs * inputs_.transpose();
	biases_gradient_ = inputs.rowwise().sum();
	inputs_gradient_ = weights_.transpose() * inputs;
}

Eigen::Index NeuronalLayer::getInputSize() const {
	return weights_.cols();
}

Eigen::Index NeuronalLayer::getNumNeurons() const {
	return weights_.rows();
}

const Eigen::MatrixXd& NeuronalLayer::getOutputs() const {
	return outputs_;
}

const Eigen::MatrixXd& NeuronalLayer::getInputsGradient() const {
	return inputs_gradient_;
}

std::ostream& operator<<(std::ostream& os, const NeuronalLayer& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "Layer\n";
	os << "  features: " << rhs.weights_.cols() << "\n";
	os << "  neurons:  " << rhs.weights_.rows() << "\n";
	os << "  outputs:\n" << rhs.outputs_.format(mat_fmt) << "\n";
	os << "  weights:\n" << rhs.weights_.format(mat_fmt) << "\n";
	os << "  biases:\n" << rhs.biases_.transpose().format(vec_fmt) << "\n";
	os << "  weights_gradient:\n"
	   << rhs.weights_gradient_.format(mat_fmt) << "\n";
	os << "  biases_gradient:\n"
	   << rhs.biases_gradient_.transpose().format(vec_fmt) << "\n";
	os << "  inputs_gradient:\n" << rhs.inputs_gradient_.format(mat_fmt);

	return os;
}
