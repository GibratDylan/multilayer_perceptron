#pragma once

#include "AActivation.hpp"

class ActivationReLU : public AActivation {
   public:
	void Forward(MatrixIn input_batch) override;
	void Backward(MatrixIn gradient_batch) override;
};
