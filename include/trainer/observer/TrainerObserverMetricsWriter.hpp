#pragma once

#include "ITrainerObserver.hpp"

#include <Eigen/Dense>

class TrainerObserverMetricsWriter : public ITrainerObserver {
   public:
	void on_epoch_end(int epoch_index, float accuracy) override;
};
