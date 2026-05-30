#pragma once

#include "AActivation.hpp"

class ActivationReLU : public AActivation {
   public:
	using InputBatch = AActivation::InputBatch;
	using OutputBatch = AActivation::OutputBatch;
	using GradientBatch = AActivation::GradientBatch;

	void Forward(const InputBatch& input_batch) override;
	void Backward(const GradientBatch& gradient_batch) override;
};
