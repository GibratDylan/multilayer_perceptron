#include "network/NetworkBuilder.hpp"

#include "config/Config.hpp"
#include "network/Network.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>

NetworkBuilder::NetworkBuilder(Config config, int64_t input_size_data)
	: config_{std::move(config)}, input_size_data_{input_size_data} {}

Network NetworkBuilder::Build() const {
	assert(config_.IsConfigValid());

	Network network{Network::GetLossFuncObj(config_.GetLossFunc())};

	for (int64_t i{}; i < config_.GetSize(); ++i) {
		int64_t input_size{i ? config_.GetNeuralLayer().at(i - 1)
							 : input_size_data_};

		network.AddLayer(
			NeuronalLayer{input_size, config_.GetNeuralLayer().at(i)},
			Network::GetActivationFuncObj(config_.GetActivationFunc().at(i)));
	}

	assert(network.GetNeuronalLayers().size() <
			   std::numeric_limits<int64_t>::max() &&
		   config_.GetSize() ==
			   static_cast<int64_t>(network.GetNeuronalLayers().size()));

	assert(network.GetNeuronalLayers().size() ==
		   network.GetActivationLayers().size());

	return network;
}
