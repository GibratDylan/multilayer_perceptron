#pragma once

#include "AActivation.hpp"

class ActivationSoftmax : public AActivation {
   public:
	void Forward(const MatrixIn& input_batch) override;
	void Backward(const MatrixIn& gradient_batch) override;
};
