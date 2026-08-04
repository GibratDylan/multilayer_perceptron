#include "layer/NeuronalLayer.hpp"

#include "types/eigen_types.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <ostream>

NeuronalLayer::NeuronalLayer(int64_t input_size, int64_t num_neurons)
	: weights_{[num_neurons, input_size] {
		  constexpr float k_weight_scale{0.1F};
		  return Matrix::Random(num_neurons, input_size) * k_weight_scale;
	  }()},
	  biases_(num_neurons) {
	biases_.setZero();
}

void NeuronalLayer::Forward(const MatrixIn& input_batch) {
	assert(input_batch.rows() == weights_.cols() && input_batch.size() > 0);

	inputs_ = input_batch;
	outputs_.noalias() = weights_ * input_batch;

	assert(outputs_.rows() == biases_.rows());

	outputs_.colwise() += biases_;

	assert(input_batch.cols() == outputs_.cols() &&
		   weights_.rows() == outputs_.rows());
}

void NeuronalLayer::Backward(const MatrixIn& gradient_batch) {
	assert(gradient_batch.rows() == weights_.rows() &&
		   gradient_batch.size() > 0);
	assert(inputs_.size() > 0 && gradient_batch.cols() == inputs_.cols());

	weights_gradient_.noalias() = gradient_batch * inputs_.transpose();
	biases_gradient_ = gradient_batch.rowwise().sum();

	assert(biases_gradient_.rows() == gradient_batch.rows());

	inputs_gradient_.noalias() = weights_.transpose() * gradient_batch;

	assert(gradient_batch.cols() == inputs_gradient_.cols() &&
		   weights_.cols() == inputs_gradient_.rows());
}

int64_t NeuronalLayer::GetInputSize() const noexcept {
	return weights_.cols();
}

int64_t NeuronalLayer::GetNumNeurons() const noexcept {
	return weights_.rows();
}

MatrixIn NeuronalLayer::GetWeightsGradient() const noexcept {
	return weights_gradient_;
}

MatrixIn NeuronalLayer::GetBiasesGradient() const noexcept {
	return biases_gradient_;
}

MatrixIn NeuronalLayer::GetOutputs() const noexcept {
	return outputs_;
}

MatrixIn NeuronalLayer::GetInputsGradient() const noexcept {
	return inputs_gradient_;
}

std::ostream& operator<<(std::ostream& os, const NeuronalLayer& rhs) {
	const Eigen::IOFormat mat_fmt{4, 0, ", ", "\n", "    [", "]"};
	const Eigen::IOFormat vec_fmt{4, 0, ", ", "\n", "    [", "]"};

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
