#pragma once

#include "layer/NeuronalLayer.hpp"
#include "activation/AActivation.hpp"
#include "loss/ALoss.hpp"
#include "types/eigen_types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class Network {
   private:
	std::vector<NeuronalLayer> neuronal_layers_;
	std::vector<std::unique_ptr<AActivation>> activation_func_;
	std::unique_ptr<ALoss> loss_func_;

	int64_t size_;

   public:
	explicit Network(std::unique_ptr<ALoss> loss_func);

	Network& AddLayer(NeuronalLayer&& neuronal_layer,
					  std::unique_ptr<AActivation>&& activation_func);

	float ForwardPass(const MatrixIn& input_batch,
					  const IntVectorIn& targets_batch);
	void BackwardPass(const IntVectorIn& targets_batch);

	const std::vector<NeuronalLayer>& GetNeuronalLayers() const;
	const std::vector<std::unique_ptr<AActivation>>& GetActivationLayers() const;
	const ALoss& GetLossFunc() const;

   private:
	void AddNeuronalLayer(NeuronalLayer&& neuronal_layer);
	void AddActivationLayer(std::unique_ptr<AActivation>&& activation_func);
};
