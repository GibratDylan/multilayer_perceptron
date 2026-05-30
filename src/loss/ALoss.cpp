#include "loss/ALoss.hpp"

#include <ostream>

LossValue ALoss::GetLoss() const {
	assert(outputs_.size() > 0);
	return outputs_.mean();
}

const ALoss::LossesBatch& ALoss::GetOutputs() const {
	return outputs_;
}

const ALoss::LogitsGradientBatch& ALoss::GetInputsGradient() const {
	return inputs_gradient_;
}

ALoss::LossFuncType ALoss::GetLossType(std::string_view str) {
	if (str == loss_func_string::kCatCrossentropy) return kCatCrossentropy;
	return kNone;
}

std::ostream& operator<<(std::ostream& os, const ALoss& rhs) {
	const Eigen::IOFormat mat_fmt(4, 0, ", ", "\n", "    [", "]");
	const Eigen::IOFormat vec_fmt(4, 0, ", ", "\n", "    [", "]");
	os << "Loss\n";
	os << "  outputs:\n" << rhs.outputs_.transpose().format(vec_fmt);
	os << "  inputs_gradient:\n"
	   << rhs.inputs_gradient_.format(mat_fmt) << "\n";
	os << "  loss_means:\n" << rhs.GetLoss();
	return os;
}
