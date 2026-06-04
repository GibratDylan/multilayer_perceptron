#include "network/NetworkBuilder.hpp"

#include "activation/AActivation.hpp"
#include "activation/ActivationReLU.hpp"
#include "activation/ActivationSoftmax.hpp"
#include "loss/ALoss.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"

#include <limits>
#include <memory>

NetworkBuilder::NetworkBuilder(const Config& config) : config_{config} {}

Network NetworkBuilder::Build() const {
	assert(config_.IsConfigValid());

	Network network{GetLossFuncObj()};

	for (int64_t i{}; i < config_.GetSize() - 1; ++i)
		network.AddLayer(NeuronalLayer{config_.GetNeuralLayer()[i],
									   config_.GetNeuralLayer()[i + 1]},
						 GetActivationFuncObj(i));

	assert(network.GetNeuronalLayers().size() <
			   std::numeric_limits<int64_t>::max() &&
		   config_.GetSize() >
			   static_cast<int64_t>(network.GetNeuronalLayers().size()));

	assert(network.GetNeuronalLayers().size() ==
		   network.GetActivationLayers().size());

	return network;
}

std::unique_ptr<ALoss> NetworkBuilder::GetLossFuncObj() const {
	switch (config_.GetLossFunc()) {
		case ALoss::LossFuncType::kCatCrossentropy:
			return std::make_unique<LossCategoricalCrossEntropy>();
		default:
			assert(false);
			return nullptr;
	}
}

std::unique_ptr<AActivation> NetworkBuilder::GetActivationFuncObj(
	int64_t index) const {

	assert(index >= 0);
	assert(config_.GetActivationFunc().size() <
			   std::numeric_limits<int64_t>::max() &&
		   index < static_cast<int64_t>(config_.GetActivationFunc().size()));

	switch (config_.GetActivationFunc()[index]) {
		case AActivation::ActivationFuncType::kRelu:
			return std::make_unique<ActivationReLU>();
		case AActivation::ActivationFuncType::kSoftmax:
			return std::make_unique<ActivationSoftmax>();
		default:
			assert(false);
			return nullptr;
	}
}
