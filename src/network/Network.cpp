#include "network/Network.hpp"

#include "layer/NeuronalLayer.hpp"
#include "types/eigen_types.hpp"

#include <limits>
#include <memory>

Network::Network(std::unique_ptr<ALoss>&& loss_func)
	: loss_func_(std::move(loss_func)), size_{} {}

Network& Network::AddLayer(NeuronalLayer&& neuronal_layer,
						   std::unique_ptr<AActivation>&& activation_func) {
	assert(activation_func != nullptr);
	assert(neuronal_layers_.size() == activation_func_.size());
	assert(neuronal_layer.GetInputSize() > 0);
	assert(neuronal_layer.GetNumNeurons() > 0);

	if (!neuronal_layers_.empty()) {
		const NeuronalLayer& previous_layer{neuronal_layers_.back()};
		assert(previous_layer.GetNumNeurons() == neuronal_layer.GetInputSize());
	}

	AddNeuronalLayer(std::move(neuronal_layer));
	AddActivationLayer(std::move(activation_func));
	assert(neuronal_layers_.size() == activation_func_.size());

	return *this;
}

void Network::ForwardPass(const MatrixIn& input_batch,
						   const IntVectorIn& targets_batch) {
	assert(size_ > 0);
	assert(loss_func_ != nullptr);
	assert(targets_batch.size() > 0 && input_batch.size() > 0);
	assert(input_batch.cols() == targets_batch.rows());

	neuronal_layers_.front().Forward(input_batch);
	activation_func_.front()->Forward(neuronal_layers_.front().GetOutputs());

	for (Eigen::Index index{1}; index < size_; ++index) {
		neuronal_layers_[index].Forward(
			activation_func_[index - 1]->GetOutputs());
		activation_func_[index]->Forward(neuronal_layers_[index].GetOutputs());
	}

	loss_func_->Forward(activation_func_.back()->GetOutputs(), targets_batch);
}

void Network::BackwardPass(const IntVectorIn& targets_batch) {
	assert(size_ > 0);
	assert(targets_batch.size() > 0);
	assert(loss_func_ != nullptr);

	loss_func_->Backward(targets_batch);

	activation_func_.back()->Backward(loss_func_->GetInputsGradient());

	for (Eigen::Index index{size_ - 1}; index > 0; --index) {
		neuronal_layers_[index].Backward(
			activation_func_[index]->GetInputsGradient());
		activation_func_[index - 1]->Backward(
			neuronal_layers_[index].GetInputsGradient());
	}

	neuronal_layers_.front().Backward(
		activation_func_.front()->GetInputsGradient());

	assert(neuronal_layers_.front().GetInputsGradient().size() > 0);
}

void Network::AddNeuronalLayer(NeuronalLayer&& neuronal_layer) {
	assert(size_ < std::numeric_limits<int64_t>::max());
	neuronal_layers_.push_back(std::move(neuronal_layer));
	++size_;
}

void Network::AddActivationLayer(
	std::unique_ptr<AActivation>&& activation_func) {
	assert(activation_func != nullptr);
	activation_func_.push_back(std::move(activation_func));
}

const std::vector<NeuronalLayer>& Network::GetNeuronalLayers() const noexcept {
	return neuronal_layers_;
}

const std::vector<std::unique_ptr<AActivation>>& Network::GetActivationLayers()
	const noexcept {
	return activation_func_;
}

const ALoss& Network::GetLossFunc() const noexcept {
	return *loss_func_;
}
