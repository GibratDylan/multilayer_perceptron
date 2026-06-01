#pragma once

#include "types/eigen_types.hpp"

#include <ostream>
#include <string_view>

namespace activation_func_string {
inline constexpr std::string_view kRelu = "relu";
inline constexpr std::string_view kSoftmax = "softmax";
}  // namespace activation_func_string

class AActivation {
   public:
	enum ActivationFuncType : uint8_t { kRelu, kSoftmax, kNone };

   protected:
	Matrix outputs_;
	Matrix inputs_;

	Matrix inputs_gradient_;

   public:
	virtual ~AActivation() = default;

	virtual void Forward(MatrixIn input_batch) = 0;
	virtual void Backward(MatrixIn gradient_batch) = 0;

	MatrixIn GetOutputs() const;
	MatrixIn GetInputsGradient() const;

	static AActivation::ActivationFuncType GetActivationType(
		std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const AActivation& rhs);
};
