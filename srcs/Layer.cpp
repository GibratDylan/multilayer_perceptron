#include "../includes/Layer.hpp"

#include <Eigen/Dense>
#include <iostream>

Layer::Layer(int nb_inputs, int nb_neurons)
	: _weights(nb_inputs, nb_neurons), _biases(nb_neurons) {
	_weights = Eigen::MatrixXd::Random(nb_inputs, nb_neurons);
	_weights *= 0.10;
	_biases.setZero();
}

void Layer::forward(const Eigen::VectorXd& inputs) {
	_outputs = _weights.transpose() * inputs;
	_outputs += _biases;
}

std::ostream& operator<<(std::ostream& os, const Layer& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "Layer\n";
	os << "  features: " << rhs._weights.rows() << "\n";
	os << "  neurons:  " << rhs._weights.cols() << "\n";
	os << "  weights:\n" << rhs._weights.format(mat_fmt) << "\n";
	os << "  biases:\n" << rhs._biases.transpose().format(vec_fmt);

	return os;
}
