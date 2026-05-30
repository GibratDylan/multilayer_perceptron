#pragma once

#include "ALoss.hpp"

class LossCategoricalCrossEntropy : public ALoss {
   public:
	using LogitsBatch = ALoss::LogitsBatch;
	using TargetsBatch = ALoss::TargetsBatch;
	using LossesBatch = ALoss::LossesBatch;
	using LogitsGradientBatch = ALoss::LogitsGradientBatch;

	void Forward(const LogitsBatch& logits_batch,
				 const TargetsBatch& targets_batch) override;
	void Backward(const TargetsBatch& targets_batch) override;
};
