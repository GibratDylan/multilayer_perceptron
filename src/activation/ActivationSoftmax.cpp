#include "activation/ActivationSoftmax.hpp"

#include "activation/AActivation.hpp"
#include "types/eigen_types.hpp"

#include <cassert>
#include <cstdint>

void ActivationSoftmax::Forward(const MatrixIn& input_batch) {
	Inputs() = input_batch;
	Outputs() = input_batch.rowwise() - input_batch.colwise().maxCoeff();
	Outputs() = Outputs().array().exp();
	Outputs().array().rowwise() /= Outputs().colwise().sum().array();

	assert(input_batch.rows() == Outputs().rows() &&
		   input_batch.cols() == Outputs().cols());
}

void ActivationSoftmax::Backward(const MatrixIn& gradient_batch) {
	InputsGradient().resizeLike(gradient_batch);
	for (int64_t i{}; i < gradient_batch.cols(); ++i) {
		Vector s{Outputs().col(i)};

		Matrix jacobian{Matrix(s.asDiagonal())};
		jacobian.noalias() -= s * s.transpose();

		InputsGradient().col(i).noalias() = jacobian * gradient_batch.col(i);
	}

	assert(InputsGradient().rows() == gradient_batch.rows() &&
		   InputsGradient().cols() == gradient_batch.cols());
}

AActivation::ActivationFuncType ActivationSoftmax::GetActivationType()
	const noexcept {
	return ActivationFuncType::kSoftmax;
}
