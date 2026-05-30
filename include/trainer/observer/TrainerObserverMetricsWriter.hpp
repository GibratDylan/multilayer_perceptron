#pragma once

#include "ITrainerObserver.hpp"

class TrainerObserverMetricsWriter : public ITrainerObserver {
   public:
	void OnEpochEnd(EpochIndex epoch_index, AccuracyScore accuracy) override;
};
