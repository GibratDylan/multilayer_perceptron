#pragma once

#include <Eigen/Dense>

class ITrainerObserver {
   public:
	virtual ~ITrainerObserver() = default;
	virtual void on_epoch_end(int epoch_index, float accuracy) = 0;
};
