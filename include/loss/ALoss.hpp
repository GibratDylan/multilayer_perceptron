#pragma once

#include "types/EigenTypes.hpp"

#include <ostream>
#include <string_view>

namespace loss_func_string {
inline constexpr std::string_view kCatCrossentropy =
	"categorical_cross_entropy";
}  // namespace loss_func_string

class ALoss {
   public:
	enum LossFuncType : uint8_t { kCatCrossentropy, kNone };

   protected:
	Vector outputs_;
	Matrix inputs_;

	Matrix inputs_gradient_;

   public:
	virtual ~ALoss() = default;

	virtual void Forward(MatrixIn logits_batch,
						 IntVectorIn targets_batch) = 0;
	virtual void Backward(IntVectorIn targets_batch) = 0;

	VectorIn GetOutputs() const;
	MatrixIn GetInputsGradient() const;
	float GetLoss() const;

	static ALoss::LossFuncType GetLossType(std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const ALoss& rhs);
};
