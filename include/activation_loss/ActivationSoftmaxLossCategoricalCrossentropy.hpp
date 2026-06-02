#pragma once

#include "activation/ActivationSoftmax.hpp"
#include "loss/LossCategoricalCrossEntropy.hpp"
#include "types/eigen_types.hpp"

#include <ostream>

class ActivationSoftmaxLossCategoricalCrossentropy {
   private:
	LossCategoricalCrossEntropy loss_{};
	ActivationSoftmax activation_{};
	Matrix outputs_;

	Matrix inputs_gradient_;

   public:
	void Forward(const MatrixIn& input_batch, const IntVectorIn& targets_batch);
	void Backward(const IntVectorIn& targets_batch);

	MatrixIn GetOutputs() const;
	MatrixIn GetInputsGradient() const;
	float GetLoss() const;

	friend std::ostream& operator<<(
		std::ostream& os,
		const ActivationSoftmaxLossCategoricalCrossentropy& rhs);
};
