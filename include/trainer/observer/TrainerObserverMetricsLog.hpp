#pragma once

#include "ITrainerObserver.hpp"

#include <cstdint>

class TrainerObserverMetricsLog : public ITrainerObserver {
   public:
	void OnEpochEnd(int64_t epoch_index, float loss, float accuracy) override;
};
