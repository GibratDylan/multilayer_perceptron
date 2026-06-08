#pragma once

#include "ALoss.hpp"
#include "types/eigen_types.hpp"

class LossCategoricalCrossEntropy : public ALoss {
   public:
	void Forward(const MatrixIn& logits_batch,
				 IntVectorIn targets_batch) override;
	void Backward(IntVectorIn targets_batch) override;
};
