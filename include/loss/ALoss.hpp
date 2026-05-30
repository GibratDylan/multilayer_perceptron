#pragma once

#include "types/EigenTypes.hpp"
#include "types/Types.hpp"

#include <ostream>
#include <string_view>

namespace loss_func_string {
inline constexpr std::string_view kCatCrossentropy =
	"categorical_cross_entropy";
}  // namespace loss_func_string

class ALoss {
   public:
	using LogitsBatch = Matrix;
	using TargetsBatch = IntVector;
	using LossesBatch = Vector;
	using LogitsGradientBatch = Matrix;

	using Logits = LogitsBatch;
	using Targets = TargetsBatch;
	using Losses = LossesBatch;
	using Gradients = LogitsGradientBatch;

	enum LossFuncType : uint8_t { kCatCrossentropy, kNone };

   protected:
	LossesBatch outputs_;
	LogitsBatch inputs_;

	LogitsGradientBatch inputs_gradient_;

   public:
	virtual ~ALoss() = default;

	virtual void Forward(const LogitsBatch& logits_batch,
						 const TargetsBatch& targets_batch) = 0;
	virtual void Backward(const TargetsBatch& targets_batch) = 0;

	const LossesBatch& GetOutputs() const;
	const LogitsGradientBatch& GetInputsGradient() const;
	LossValue GetLoss() const;

	static ALoss::LossFuncType GetLossType(std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const ALoss& rhs);
};
