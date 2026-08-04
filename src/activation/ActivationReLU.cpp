#include "activation/ActivationReLU.hpp"

#include "activation/AActivation.hpp"
#include "types/eigen_types.hpp"

#include <cassert>
#include <cstdint>

void ActivationReLU::Forward(const MatrixIn& input_batch) {
	Inputs() = input_batch;
	Outputs() = input_batch.cwiseMax(0.F);

	assert(input_batch.rows() == Outputs().rows() &&
		   input_batch.cols() == Outputs().cols());
	assert(Outputs().minCoeff() >= 0.F);
}

void ActivationReLU::Backward(const MatrixIn& gradient_batch) {
	InputsGradient() = gradient_batch;
	for (int64_t i{}, size{InputsGradient().size()}; i < size; ++i) {
		Matrix::Scalar temporary{Inputs().coeff(i)};

		if (temporary < 0) InputsGradient().coeffRef(i) = 0.F;
	}

	assert(InputsGradient().rows() == gradient_batch.rows() &&
		   InputsGradient().cols() == gradient_batch.cols());
}

AActivation::ActivationFuncType ActivationReLU::GetActivationType()
	const noexcept {
	return ActivationFuncType::kRelu;
}
