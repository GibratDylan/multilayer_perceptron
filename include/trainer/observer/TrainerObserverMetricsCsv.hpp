#pragma once

#include "ITrainerObserver.hpp"
#include "data/csv.hpp"

#include <cstdint>

class TrainerObserverMetricsCsv : public ITrainerObserver {
   private:
	csv::WriteToCsv csv_writer_{"metrics.csv", "epoch_index,loss,accuracy"};

   public:
	void OnEpochEnd(int64_t epoch_index, float loss, float accuracy) override;
};
