#pragma once

#include "types/EigenTypes.hpp"

#include <ostream>
#include <string_view>

namespace activation_func_string {
inline constexpr std::string_view kRelu = "relu";
inline constexpr std::string_view kSoftmax = "softmax";
}  // namespace activation_func_string

class AActivation {
   public:
	using InputBatch = Matrix;
	using OutputBatch = Matrix;
	using GradientBatch = Matrix;

	using Inputs = InputBatch;
	using Outputs = OutputBatch;
	using Gradients = GradientBatch;

	enum ActivationFuncType : uint8_t { kRelu, kSoftmax, kNone };

   protected:
	OutputBatch outputs_;
	InputBatch inputs_;

	GradientBatch inputs_gradient_;

   public:
	virtual ~AActivation() = default;

	virtual void Forward(const InputBatch& input_batch) = 0;
	virtual void Backward(const GradientBatch& gradient_batch) = 0;

	const OutputBatch& GetOutputs() const;
	const GradientBatch& GetInputsGradient() const;

	static AActivation::ActivationFuncType GetActivationType(
		std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const AActivation& rhs);
};
