#include "activation/AActivation.hpp"

#include "error/Result.hpp"
#include "error/error.hpp"
#include "types/eigen_types.hpp"

#include <ostream>
#include <string_view>

MatrixIn AActivation::GetOutputs() const noexcept {
	return Outputs();
}

MatrixIn AActivation::GetInputsGradient() const noexcept {
	return InputsGradient();
}

Result<AActivation::ActivationFuncType, ActivationFuncError> AActivation::GetActivationType(
	std::string_view str) noexcept {
	if (str == activation_func_string::kRelu) return Result<ActivationFuncType, ActivationFuncError>{ActivationFuncType::kRelu};
	if (str == activation_func_string::kSoftmax)
		return Result<ActivationFuncType, ActivationFuncError>{ActivationFuncType::kSoftmax};
	return Result<ActivationFuncType, ActivationFuncError>{ActivationFuncError::kNotValidActivationFunc};
}

std::ostream& operator<<(std::ostream& os, const AActivation& rhs) {
	const Eigen::IOFormat mat_fmt{4, 0, ", ", "\n", "    [", "]"};

	os << "Activation\n";
	os << "  outputs:\n" << rhs.GetOutputs().format(mat_fmt) << "\n";
	os << "  inputs_gradient:\n" << rhs.GetInputsGradient().format(mat_fmt);

	return os;
}
