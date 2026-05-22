#include "Layer.hpp"

#include <ostream>

#include <Eigen/Dense>

// Do i really need constructor ?
Layer::Layer(int input_size, int num_neurons)
	: weights_{Eigen::MatrixXd::Random(num_neurons, input_size)},
	  biases_(num_neurons) {
	weights_ *= 0.10;
	biases_.setZero();
}

void Layer::forward(const Eigen::MatrixXd& inputs) {
	assert(inputs.rows() == weights_.cols() && inputs.size() > 0);

	inputs_ = inputs;
	outputs_ = weights_ * inputs;

	assert(outputs_.rows() == biases_.rows());

	outputs_.colwise() += biases_;

	assert(inputs.cols() == outputs_.cols() &&
		   weights_.rows() == outputs_.rows());
}

void Layer::backward(const Eigen::MatrixXd& inputs) {
	weights_gradient_ = inputs * inputs_.transpose();
	biases_gradient_ = inputs.rowwise().sum();
	inputs_gradient_ = weights_.transpose() * inputs;
}

const Eigen::MatrixXd& Layer::getOutputs() const {
	return outputs_;
}

const Eigen::MatrixXd& Layer::getInputsGradient() const {
	return inputs_gradient_;
}

std::ostream& operator<<(std::ostream& os, const Layer& rhs) {
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
