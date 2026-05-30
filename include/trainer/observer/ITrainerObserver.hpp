#pragma once

#include "types/Types.hpp"

class ITrainerObserver {
   public:
	virtual ~ITrainerObserver() = default;
	virtual void OnEpochEnd(EpochIndex epoch_index, AccuracyScore accuracy) = 0;
};
