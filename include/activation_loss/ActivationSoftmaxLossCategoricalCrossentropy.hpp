#pragma once

#include "activation/ActivationSoftmax.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "types/EigenTypes.hpp"
#include "types/Types.hpp"

#include <ostream>

class ActivationSoftmaxLossCategoricalCrossentropy {
   public:
	using InputBatch = Matrix;
	using OutputBatch = Matrix;
	using TargetsBatch = IntVector;
	using GradientBatch = Matrix;

	using Inputs = InputBatch;
	using Outputs = OutputBatch;
	using Targets = TargetsBatch;
	using Gradients = GradientBatch;

   private:
	LossCategoricalCrossEntropy loss_ = LossCategoricalCrossEntropy();
	ActivationSoftmax activation_ = ActivationSoftmax();
	OutputBatch outputs_;

	GradientBatch inputs_gradient_;

   public:
	LossValue Forward(const InputBatch& input_batch,
					  const TargetsBatch& targets_batch);
	void Backward(const TargetsBatch& targets_batch);

	const OutputBatch& GetOutputs() const;
	const GradientBatch& GetInputsGradient() const;

	friend std::ostream& operator<<(
		std::ostream& os,
		const ActivationSoftmaxLossCategoricalCrossentropy& rhs);
};
