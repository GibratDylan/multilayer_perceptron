#pragma once

#include <cstdint>

class ITrainerObserver {
   public:
	virtual ~ITrainerObserver() = default;
	virtual void OnEpochEnd(int64_t epoch_index, float accuracy) = 0;
};
