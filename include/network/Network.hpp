#pragma once

#include "NeuronalLayer.hpp"
#include "activation/AActivation.hpp"
#include "loss/ALoss.hpp"

#include <memory>
#include <vector>

class Network {
   private:
	std::vector<NeuronalLayer> neuronal_layers_;
	std::vector<std::unique_ptr<AActivation>> activation_layers_;
	std::unique_ptr<ALoss> loss_layer_;

	size_t size_{};

   public:
	Network(std::unique_ptr<ALoss> loss_layer);

	Network& addLayer(NeuronalLayer&& neuronal_layer,
					  std::unique_ptr<AActivation>&& activation_layer);

	double forwardPass(const Eigen::MatrixXd& inputs,
					 const Eigen::VectorXi& target_inputs);
	void backwardPass(const Eigen::MatrixXd& inputs);

	const NeuronalLayer& getNeuronalLayer() const;
	const AActivation& getActivationLayer() const;
	const ALoss& getLossLayer() const;

	// void operator++();

   private:
	void addNeuronalLayer(NeuronalLayer&& neuronal_layer);
	void addActivationLayer(std::unique_ptr<AActivation>&& activation_layer);
};
