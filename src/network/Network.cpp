#include "network/Network.hpp"

#include <limits>
#include <memory>

#include <Eigen/Dense>

Network::Network(std::unique_ptr<ALoss> loss_func)
	: loss_func_(std::move(loss_func)) {}

Network& Network::addLayer(NeuronalLayer&& neuronal_layer,
						   std::unique_ptr<AActivation>&& activation_func) {
	assert(activation_func != nullptr);
	assert(neuronal_layers_.size() == activation_func_.size());
	assert(neuronal_layer.getInputSize() > 0);
	assert(neuronal_layer.getNumNeurons() > 0);

	if (!neuronal_layers_.empty()) {
		const NeuronalLayer& previous_layer = neuronal_layers_.back();
		assert(previous_layer.getNumNeurons() == neuronal_layer.getInputSize());
	}

	addNeuronalLayer(std::move(neuronal_layer));
	addActivationLayer(std::move(activation_func));
	assert(neuronal_layers_.size() == activation_func_.size());

	return *this;
}

double Network::forwardPass(const Eigen::MatrixXd& inputs,
							const Eigen::VectorXi& target_inputs) {
	assert(size_ > 0);
	assert(inputs.cols() == target_inputs.rows() &&
		   target_inputs.maxCoeff() <= inputs.rows());
	assert(inputs.size() > 0 && target_inputs.size() > 0);

	std::reference_wrapper<const Eigen::MatrixXd> current = inputs;

	for (size_t index = 0; index < size_; ++index) {
		neuronal_layers_[index].forward(current);
		activation_func_[index]->forward(neuronal_layers_[index].getOutputs());
		current = activation_func_[index]->getOutputs();
	}

	loss_func_->forward(activation_func_[size_ - 1]->getOutputs(),
						target_inputs);

	return loss_func_->getLoss();
}

// DOIT ETRE IMPLEMENTER
void Network::backwardPass(const Eigen::MatrixXd& inputs) {
	(void)inputs;
}

void Network::addNeuronalLayer(NeuronalLayer&& neuronal_layer) {
	assert(size_ < std::numeric_limits<size_t>::max());
	neuronal_layers_.push_back(std::move(neuronal_layer));
	++size_;
}

void Network::addActivationLayer(
	std::unique_ptr<AActivation>&& activation_func) {
	assert(activation_func != nullptr);
	activation_func_.push_back(std::move(activation_func));
}

// const NeuronalLayer& Network::getNeuronalLayer() const {
// 	return neuronal_layers_[index_];
// }

// const AActivation& Network::getActivationLayer() const {
// 	return *(activation_func_[index_]);
// }

// const ALoss& Network::getLossLayer() const { return *loss_func_; }

// void Network::operator++() {
// 	assert(index_ < std::numeric_limits<size_t>::max() && index_ < size_);
// 	++index_;
// }
