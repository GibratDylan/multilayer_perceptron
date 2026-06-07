#include "trainer/observer/TrainerObserverMetricsCsv.hpp"

#include "data/csv.hpp"

#include <cstdint>

void TrainerObserverMetricsCsv::OnEpochEnd(int64_t epoch_index, float loss,
										   float accuracy) {
	csv::WriteToCsv("metrics.csv", "epoch_index,loss,accuracy", epoch_index,
					loss, accuracy);
}
