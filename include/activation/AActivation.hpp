#pragma once

#include "types/eigen_types.hpp"

#include <cstdint>
#include <ostream>
#include <string_view>

namespace activation_func_string {
inline constexpr std::string_view kRelu{"relu"};
inline constexpr std::string_view kSoftmax{"softmax"};
}  // namespace activation_func_string

class AActivation {
   public:
	enum class ActivationFuncType : uint8_t { kRelu, kSoftmax, kNone };

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

	Matrix& Outputs() noexcept { return outputs_; }

	const Matrix& Outputs() const noexcept { return outputs_; }

	Matrix& Inputs() noexcept { return inputs_; }

	const Matrix& Inputs() const noexcept { return inputs_; }

	Matrix& InputsGradient() noexcept { return inputs_gradient_; }

	const Matrix& InputsGradient() const noexcept { return inputs_gradient_; }

   private:
	Matrix outputs_{};
	Matrix inputs_{};

	Matrix inputs_gradient_{};
};
