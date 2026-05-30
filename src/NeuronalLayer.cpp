#include "NeuronalLayer.hpp"

#include <ostream>

NeuronalLayer::NeuronalLayer(InputSize input_size, NeuronCount num_neurons)
	: weights_{Matrix::Random(static_cast<Index>(num_neurons),
							  static_cast<Index>(input_size))},
	  biases_(static_cast<Index>(num_neurons)) {
	weights_ *= 0.10;
	biases_.setZero();
}

void NeuronalLayer::Forward(const InputBatch& input_batch) {
	assert(input_batch.rows() == weights_.cols() && input_batch.size() > 0);

	inputs_ = input_batch;
	outputs_ = weights_ * input_batch;

	assert(outputs_.rows() == biases_.rows());

	outputs_.colwise() += biases_;

	assert(input_batch.cols() == outputs_.cols() &&
		   weights_.rows() == outputs_.rows());
}

void NeuronalLayer::Backward(const GradientBatch& gradient_batch) {
	weights_gradient_ = gradient_batch * inputs_.transpose();
	biases_gradient_ = gradient_batch.rowwise().sum();
	inputs_gradient_ = weights_.transpose() * gradient_batch;
}

Index NeuronalLayer::GetInputSize() const {
	return weights_.cols();
}

Index NeuronalLayer::GetNumNeurons() const {
	return weights_.rows();
}

const NeuronalLayer::OutputBatch& NeuronalLayer::GetOutputs() const {
	return outputs_;
}

const NeuronalLayer::GradientBatch& NeuronalLayer::GetInputsGradient() const {
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
