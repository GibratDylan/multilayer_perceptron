#include "loss/ALoss.hpp"

#include "error/Result.hpp"
#include "error/error.hpp"
#include "types/eigen_types.hpp"

#include <cassert>
#include <ostream>
#include <string_view>

float ALoss::GetLoss() const noexcept {
	assert(Outputs().size() > 0);
	return Outputs().mean();
}

VectorIn ALoss::GetOutputs() const noexcept {
	return Outputs();
}

MatrixIn ALoss::GetInputsGradient() const noexcept {
	return InputsGradient();
}

Result<ALoss::LossFuncType, LossFuncError> ALoss::LossType(
	std::string_view str) noexcept {
	if (str == loss_func_string::kCatCrossentropy)
		return Result<ALoss::LossFuncType, LossFuncError>{
			LossFuncType::kCatCrossentropy};
	return Result<ALoss::LossFuncType, LossFuncError>{
		LossFuncError::kNotValidLoss};
}

std::ostream& operator<<(std::ostream& os, const ALoss& rhs) {
	const Eigen::IOFormat mat_fmt{4, 0, ", ", "\n", "    [", "]"};
	const Eigen::IOFormat vec_fmt{4, 0, ", ", "\n", "    [", "]"};
	os << "Loss\n";
	os << "  outputs:\n" << rhs.GetOutputs().transpose().format(vec_fmt);
	os << "  inputs_gradient:\n"
	   << rhs.GetInputsGradient().format(mat_fmt) << "\n";
	os << "  loss_means:\n" << rhs.GetLoss();
	return os;
}
