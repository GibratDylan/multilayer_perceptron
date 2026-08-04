#pragma once

#include "AActivation.hpp"
#include "types/eigen_types.hpp"

class ActivationSoftmax : public AActivation {
   public:
	void Forward(const MatrixIn& input_batch) override;
	void Backward(const MatrixIn& gradient_batch) override;
	ActivationFuncType GetActivationType() const noexcept override;
};
