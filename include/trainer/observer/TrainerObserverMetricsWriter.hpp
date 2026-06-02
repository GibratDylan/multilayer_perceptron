#pragma once

#include "ITrainerObserver.hpp"

#include <cstdint>

class TrainerObserverMetricsWriter : public ITrainerObserver {
   public:
	void OnEpochEnd(int64_t epoch_index, float accuracy) override;
};
