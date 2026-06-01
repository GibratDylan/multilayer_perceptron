#pragma once

#include "activation/ActivationSoftmax.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "types/eigen_types.hpp"

#include <ostream>

class ActivationSoftmaxLossCategoricalCrossentropy {
   private:
	LossCategoricalCrossEntropy loss_ = LossCategoricalCrossEntropy();
	ActivationSoftmax activation_ = ActivationSoftmax();
	Matrix outputs_;

	Matrix inputs_gradient_;

   public:
	void Forward(MatrixIn input_batch, IntVectorIn targets_batch);
	void Backward(IntVectorIn targets_batch);

	MatrixIn GetOutputs() const;
	MatrixIn GetInputsGradient() const;
	float GetLoss() const;

	friend std::ostream& operator<<(
		std::ostream& os,
		const ActivationSoftmaxLossCategoricalCrossentropy& rhs);
};
