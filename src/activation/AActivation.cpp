#include "activation/AActivation.hpp"

#include "types/eigen_types.hpp"

#include <ostream>
#include <string_view>

MatrixIn AActivation::GetOutputs() const noexcept {
	return Outputs();
}

MatrixIn AActivation::GetInputsGradient() const noexcept {
	return InputsGradient();
}

AActivation::ActivationFuncType AActivation::GetActivationType(
	std::string_view str) {
	if (str == activation_func_string::kRelu) return ActivationFuncType::kRelu;
	if (str == activation_func_string::kSoftmax)
		return ActivationFuncType::kSoftmax;
	return ActivationFuncType::kNone;
}

std::ostream& operator<<(std::ostream& os, const AActivation& rhs) {
	const Eigen::IOFormat mat_fmt{4, 0, ", ", "\n", "    [", "]"};

	os << "Activation\n";
	os << "  outputs:\n" << rhs.GetOutputs().format(mat_fmt) << "\n";
	os << "  inputs_gradient:\n" << rhs.GetInputsGradient().format(mat_fmt);

	return os;
}
