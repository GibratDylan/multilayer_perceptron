#include "network/Network.hpp"

#include <functional>
#include <limits>
#include <memory>

Network::Network(std::unique_ptr<ALoss> loss_func)
	: loss_func_(std::move(loss_func)) {}

Network& Network::AddLayer(NeuronalLayer&& neuronal_layer,
						   std::unique_ptr<AActivation>&& activation_func) {
	assert(activation_func != nullptr);
	assert(neuronal_layers_.size() == activation_func_.size());
	assert(neuronal_layer.GetInputSize() > 0);
	assert(neuronal_layer.GetNumNeurons() > 0);

	if (!neuronal_layers_.empty()) {
		const NeuronalLayer& previous_layer = neuronal_layers_.back();
		assert(previous_layer.GetNumNeurons() == neuronal_layer.GetInputSize());
	}

	AddNeuronalLayer(std::move(neuronal_layer));
	AddActivationLayer(std::move(activation_func));
	assert(neuronal_layers_.size() == activation_func_.size());

	return *this;
}

LossValue Network::ForwardPass(const InputBatch& input_batch,
							   const TargetsBatch& targets_batch) {
	assert(size_ > 0);
	assert(input_batch.cols() == targets_batch.rows() &&
		   targets_batch.maxCoeff() <= input_batch.rows());
	assert(input_batch.size() > 0 && targets_batch.size() > 0);

	std::reference_wrapper<const InputBatch> current = input_batch;

	for (LayerIndex index = 0; index < size_; ++index) {
		neuronal_layers_[index].Forward(current);
		activation_func_[index]->Forward(neuronal_layers_[index].GetOutputs());
		current = activation_func_[index]->GetOutputs();
	}

	loss_func_->Forward(activation_func_[size_ - 1]->GetOutputs(),
						targets_batch);

	return loss_func_->GetLoss();
}

// DOIT ETRE IMPLEMENTER
void Network::BackwardPass(const InputBatch& input_batch) {
	(void)input_batch;
}

void Network::AddNeuronalLayer(NeuronalLayer&& neuronal_layer) {
	assert(size_ < std::numeric_limits<LayerCount>::max());
	neuronal_layers_.push_back(std::move(neuronal_layer));
	++size_;
}

void Network::AddActivationLayer(
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
