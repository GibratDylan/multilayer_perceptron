#pragma once

#include "types/eigen_types.hpp"

#include <cstdint>
#include <ostream>
#include <string_view>

namespace loss_func_string {
inline constexpr std::string_view kCatCrossentropy{"categorical_cross_entropy"};
}  // namespace loss_func_string

class ALoss {
   public:
	enum class LossFuncType : uint8_t { kCatCrossentropy, kNone };

   public:
	ALoss(const ALoss&) = delete;
	ALoss(ALoss&&) noexcept = delete;
	ALoss& operator=(const ALoss&) = delete;
	ALoss& operator=(ALoss&&) noexcept = delete;
	virtual ~ALoss() = default;

	virtual void Forward(const MatrixIn& logits_batch,
						 IntVectorIn targets_batch) = 0;
	virtual void Backward(IntVectorIn targets_batch) = 0;

	VectorIn GetOutputs() const noexcept;
	MatrixIn GetInputsGradient() const noexcept;
	float GetLoss() const noexcept;

	static ALoss::LossFuncType GetLossType(std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const ALoss& rhs);

   protected:
	ALoss() = default;

	Vector& Outputs() noexcept { return outputs_; }

	const Vector& Outputs() const noexcept { return outputs_; }

	Matrix& Inputs() noexcept { return inputs_; }

	const Matrix& Inputs() const noexcept { return inputs_; }

	Matrix& InputsGradient() noexcept { return inputs_gradient_; }

	const Matrix& InputsGradient() const noexcept { return inputs_gradient_; }

	static constexpr float kClampEpsilon{1e-7F};
	
   private:
	Vector outputs_{};
	Matrix inputs_{};

	Matrix inputs_gradient_{};
};
