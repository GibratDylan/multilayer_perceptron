#pragma once

#include "error/Result.hpp"
#include "error/error.hpp"
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
	enum class ActivationFuncType : uint8_t { kRelu, kSoftmax };

   public:
	explicit AActivation(const AActivation&) = delete;
	explicit AActivation(AActivation&&) noexcept = delete;
	AActivation& operator=(const AActivation&) = delete;
	AActivation& operator=(AActivation&&) noexcept = delete;
	virtual ~AActivation() = default;

	virtual void Forward(const MatrixIn& input_batch) = 0;
	virtual void Backward(const MatrixIn& gradient_batch) = 0;
	virtual ActivationFuncType GetActivationType() const noexcept = 0;

	MatrixIn GetOutputs() const noexcept;
	MatrixIn GetInputsGradient() const noexcept;
	static Result<ActivationFuncType, ActivationFuncError> ActivationType(
		std::string_view str) noexcept;

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
