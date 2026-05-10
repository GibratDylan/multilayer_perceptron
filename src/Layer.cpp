#include "Layer.hpp"

#include <ostream>

#include <Eigen/Dense>

// Do i really need constructor ?
Layer::Layer(int size_inputs, int nb_neurons)
	: _weights{Eigen::MatrixXd::Random(nb_neurons, size_inputs)},
	  _biases(nb_neurons) {
	_weights *= 0.10;
	_biases.setZero();
}

void Layer::forward(const Eigen::MatrixXd& inputs) {
	assert(inputs.rows() == _weights.cols() && inputs.size() > 0);

	_outputs = _weights * inputs;

	assert(_outputs.rows() == _biases.rows());
	_outputs.colwise() += _biases;

	assert(inputs.cols() == _outputs.cols() &&
		   _weights.rows() == _outputs.rows());
}

const Eigen::MatrixXd& Layer::getOutputs() const {
	return _outputs;
}

std::ostream& operator<<(std::ostream& os, const Layer& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "Layer\n";
	os << "  features: " << rhs._weights.cols() << "\n";
	os << "  neurons:  " << rhs._weights.rows() << "\n";
	os << "  weights:\n" << rhs._weights.format(mat_fmt) << "\n";
	os << "  biases:\n" << rhs._biases.transpose().format(vec_fmt);

	return os;
}
