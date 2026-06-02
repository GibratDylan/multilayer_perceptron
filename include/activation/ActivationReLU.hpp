#pragma once

#include "AActivation.hpp"

class ActivationReLU : public AActivation {
   public:
	void Forward(const MatrixIn& input_batch) override;
	void Backward(const MatrixIn& gradient_batch) override;
};
