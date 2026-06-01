#include "network/Network.hpp"

#include "types/eigen_types.hpp"

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

float Network::ForwardPass(MatrixIn input_batch, IntVectorIn targets_batch) {
	assert(size_ > 0);
	assert(targets_batch.size() > 0 && input_batch.size() > 0);
	assert(input_batch.cols() == targets_batch.rows());

	neuronal_layers_[0].Forward(input_batch);
	activation_func_[0]->Forward(neuronal_layers_[0].GetOutputs());

	for (Eigen::Index index = 1; index < size_; ++index) {
		neuronal_layers_[index].Forward(
			activation_func_[index - 1]->GetOutputs());
		activation_func_[index]->Forward(neuronal_layers_[index].GetOutputs());
	}

	loss_func_->Forward(activation_func_[size_ - 1]->GetOutputs(),
						targets_batch);
	return loss_func_->GetLoss();
}

// DOIT ETRE IMPLEMENTER
void Network::BackwardPass(MatrixIn input_batch) {
	(void)input_batch;
}

void Network::AddNeuronalLayer(NeuronalLayer&& neuronal_layer) {
	assert(size_ < std::numeric_limits<int>::max());
	neuronal_layers_.push_back(std::move(neuronal_layer));
	++size_;
}

void Network::AddActivationLayer(
	std::unique_ptr<AActivation>&& activation_func) {
	assert(activation_func != nullptr);
	activation_func_.push_back(std::move(activation_func));
}
