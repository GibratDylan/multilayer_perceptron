#pragma once

#include "types/EigenTypes.hpp"
#include "types/Types.hpp"

#include <ostream>

class NeuronalLayer {
   public:
	using InputBatch = Matrix;
	using OutputBatch = Matrix;
	using GradientBatch = Matrix;
	using Weights = Matrix;
	using Biases = Vector;

	using Inputs = InputBatch;
	using Outputs = OutputBatch;
	using Gradients = GradientBatch;

   private:
	Weights weights_;
	Biases biases_;
	OutputBatch outputs_;
	InputBatch inputs_;

	Weights weights_gradient_;
	Biases biases_gradient_;
	GradientBatch inputs_gradient_;

   public:
	explicit NeuronalLayer(InputSize input_size, NeuronCount num_neurons);
	NeuronalLayer(const NeuronalLayer& other) = default;
	NeuronalLayer& operator=(const NeuronalLayer& other) = default;
	NeuronalLayer(NeuronalLayer&& other) = default;
	NeuronalLayer& operator=(NeuronalLayer&& other) = default;

	void Forward(const InputBatch& input_batch);
	void Backward(const GradientBatch& gradient_batch);
	Index GetInputSize() const;
	Index GetNumNeurons() const;
	const OutputBatch& GetOutputs() const;
	const GradientBatch& GetInputsGradient() const;

	friend std::ostream& operator<<(std::ostream& os, const NeuronalLayer& rhs);
};
