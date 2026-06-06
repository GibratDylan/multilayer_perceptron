#pragma once

#include "types/eigen_types.hpp"

#include <ostream>
#include <string_view>

namespace activation_func_string {
inline constexpr std::string_view kRelu{"relu"};
inline constexpr std::string_view kSoftmax{"softmax"};
}  // namespace activation_func_string

class AActivation {
   public:
	enum class ActivationFuncType : uint8_t { kRelu, kSoftmax, kNone };

   protected:
	Matrix outputs_{};
	Matrix inputs_{};

	Matrix inputs_gradient_{};

   public:
	AActivation(const AActivation&) = delete;
	AActivation(AActivation&&) noexcept = delete;
	AActivation& operator=(const AActivation&) = delete;
	AActivation& operator=(AActivation&&) noexcept = delete;
	virtual ~AActivation() = default;

	virtual void Forward(const MatrixIn& input_batch) = 0;
	virtual void Backward(const MatrixIn& gradient_batch) = 0;

	MatrixIn GetOutputs() const noexcept;
	MatrixIn GetInputsGradient() const noexcept;

	static AActivation::ActivationFuncType GetActivationType(
		std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const AActivation& rhs);

   protected:
	AActivation() = default;
};
