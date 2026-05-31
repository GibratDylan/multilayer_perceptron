#pragma once

#include "NeuronalLayer.hpp"
#include "activation/AActivation.hpp"
#include "loss/ALoss.hpp"
#include "types/EigenTypes.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class Network {
   private:
	std::vector<NeuronalLayer> neuronal_layers_;
	std::vector<std::unique_ptr<AActivation>> activation_func_;
	std::unique_ptr<ALoss> loss_func_;

	int64_t size_{};

   public:
	explicit Network(std::unique_ptr<ALoss> loss_func);

	Network& AddLayer(NeuronalLayer&& neuronal_layer,
					  std::unique_ptr<AActivation>&& activation_func);

	float ForwardPass(MatrixIn input_batch, IntVectorIn targets_batch);
	void BackwardPass(MatrixIn input_batch);

	const NeuronalLayer& GetNeuronalLayer() const;
	const AActivation& GetActivationLayer() const;
	const ALoss& GetLossLayer() const;

   private:
	void AddNeuronalLayer(NeuronalLayer&& neuronal_layer);
	void AddActivationLayer(std::unique_ptr<AActivation>&& activation_func);
};
