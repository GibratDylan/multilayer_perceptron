#pragma once

#include "ALoss.hpp"

class LossCategoricalCrossEntropy : public ALoss {
   public:
	void Forward(MatrixIn logits_batch, IntVectorIn targets_batch) override;
	void Backward(IntVectorIn targets_batch) override;
};
