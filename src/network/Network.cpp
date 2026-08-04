#include "network/Network.hpp"

#include "activation/AActivation.hpp"
#include "activation/ActivationReLU.hpp"
#include "activation/ActivationSoftmax.hpp"
#include "layer/NeuronalLayer.hpp"
#include "loss/ALoss.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "types/eigen_types.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

Network::Network(std::unique_ptr<ALoss>&& loss_func)
	: loss_func_(std::move(loss_func)) {}

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
		const std::size_t layer_index{static_cast<std::size_t>(index)};
		neuronal_layers_.at(layer_index)
			.Forward(activation_func_.at(layer_index - 1)->GetOutputs());
		activation_func_.at(layer_index)
			->Forward(neuronal_layers_.at(layer_index).GetOutputs());
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
		const std::size_t layer_index{static_cast<std::size_t>(index)};
		neuronal_layers_.at(layer_index)
			.Backward(activation_func_.at(layer_index)->GetInputsGradient());
		activation_func_.at(layer_index - 1)
			->Backward(neuronal_layers_.at(layer_index).GetInputsGradient());
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

void Network::Serialize(std::string_view filename) const {
	std::ofstream file{std::string{filename}, std::ios::binary};
	if (!file.is_open())
		throw std::runtime_error("Error: Failed to open and serialize network");

	// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
	ALoss::LossFuncType loss_type{this->loss_func_->GetLossType()};
	file.write(reinterpret_cast<const char*>(&loss_type),
			   sizeof(ALoss::LossFuncType));

	uint64_t size_activation_type{this->activation_func_.size()};
	file.write(reinterpret_cast<const char*>(&size_activation_type),
			   sizeof(uint64_t));

	for (const auto& activation_func : activation_func_) {
		AActivation::ActivationFuncType activation_type{
			activation_func->GetActivationType()};
		file.write(reinterpret_cast<const char*>(&activation_type),
				   sizeof(AActivation::ActivationFuncType));
	}
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
}

Network Network::Deserialize(std::string_view filename) {
	std::ifstream file{std::string{filename}, std::ios::binary};
	if (!file.is_open())
		throw std::runtime_error(
			"Error: Failed to open and deserialize network");

	// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
	ALoss::LossFuncType loss_type{};
	file.read(reinterpret_cast<char*>(&loss_type), sizeof(ALoss::LossFuncType));
	Network network{GetLossFuncObj(loss_type)};

	uint64_t size_activation_type{};
	file.read(reinterpret_cast<char*>(&size_activation_type), sizeof(uint64_t));
	std::cout << size_activation_type << '\n';

	std::vector<AActivation::ActivationFuncType> activation_types{};
	activation_types.reserve(size_activation_type);
	for (uint64_t i{}; i < size_activation_type; ++i) {
		AActivation::ActivationFuncType activation_type{};
		file.read(reinterpret_cast<char*>(&activation_type),
				  sizeof(AActivation::ActivationFuncType));
		activation_types.push_back(activation_type);
	}
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

	return network;
}

std::unique_ptr<ALoss> Network::GetLossFuncObj(ALoss::LossFuncType loss_type) {
	switch (loss_type) {
		case ALoss::LossFuncType::kCatCrossentropy:
			return std::make_unique<LossCategoricalCrossEntropy>();
		default:
			assert(false);
			throw std::runtime_error("Loss function not valid");
	}
}

std::unique_ptr<AActivation> Network::GetActivationFuncObj(
	AActivation::ActivationFuncType activation_type) {
	switch (activation_type) {
		case AActivation::ActivationFuncType::kRelu:
			return std::make_unique<ActivationReLU>();
		case AActivation::ActivationFuncType::kSoftmax:
			return std::make_unique<ActivationSoftmax>();
		default:
			assert(false);
			throw std::runtime_error("Activation function not valid");
	}
}
