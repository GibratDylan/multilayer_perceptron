#include "activation/AActivation.hpp"

#include <ostream>

const AActivation::OutputBatch& AActivation::GetOutputs() const {
	return outputs_;
}

const AActivation::GradientBatch& AActivation::GetInputsGradient() const {
	return inputs_gradient_;
}

AActivation::ActivationFuncType AActivation::GetActivationType(
	std::string_view str) {
	if (str == activation_func_string::kRelu) return kRelu;
	if (str == activation_func_string::kSoftmax) return kSoftmax;
	return kNone;
}

std::ostream& operator<<(std::ostream& os, const AActivation& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");

	os << "Activation\n";
	os << "  outputs:\n" << rhs.GetOutputs().format(mat_fmt) << "\n";
	os << "  inputs_gradient:\n" << rhs.GetInputsGradient().format(mat_fmt);

	return os;
}
