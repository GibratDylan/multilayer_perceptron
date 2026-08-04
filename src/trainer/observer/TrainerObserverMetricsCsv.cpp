#include "trainer/observer/TrainerObserverMetricsCsv.hpp"

#include "data/csv.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

void TrainerObserverMetricsCsv::OnEpochEnd(int64_t epoch_index, float loss,
										   float accuracy) {
	if (!csv_writer_) {
		throw std::runtime_error(
			std::string(csv::CsvErrorMessage(csv_writer_.error())));
	}

	auto write_result{csv_writer_.value().Write(epoch_index, loss, accuracy)};
	if (!write_result) {
		throw std::runtime_error(
			std::string(csv::CsvErrorMessage(write_result.error())));
	}
}
