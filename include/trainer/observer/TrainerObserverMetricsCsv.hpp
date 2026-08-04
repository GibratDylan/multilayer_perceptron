#pragma once

#include "ITrainerObserver.hpp"
#include "data/csv.hpp"

#include <cstdint>

class TrainerObserverMetricsCsv : public ITrainerObserver {
   private:
	Result<csv::WriteToCsv, CsvError> csv_writer_{
		csv::WriteToCsv::Create("metrics.csv", "epoch_index,loss,accuracy")};

   public:
	void OnEpochEnd(int64_t epoch_index, float loss, float accuracy) override;
};
