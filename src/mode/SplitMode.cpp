#include "mode/SplitMode.hpp"

#include "data/Dataset.hpp"
#include "data/csv.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>

SplitMode::SplitMode(std::string_view split_data_path, float split_ratio)
	: split_data_path_{split_data_path}, split_ratio_{split_ratio} {}

void SplitMode::Run() const {
	auto data_result{csv::CsvLoader(std::string{split_data_path_})};
	if (!data_result) {
		throw std::runtime_error(
			std::string{csv::CsvErrorMessage(data_result.error())});
	}

	Dataset data{std::move(data_result.value())};

	auto pair{csv::DatasetSplit(data, split_ratio_)};

	std::string path{split_data_path_};

	size_t pos{path.find_first_of('/') + 1};
	if (pos == std::string::npos) pos = 0;

	auto training_result{csv::CsvDumper(
		path.substr().insert(pos, "predictive_"), "", pair.first)};
	if (!training_result) {
		throw std::runtime_error(
			std::string{csv::CsvErrorMessage(training_result.error())});
	}

	auto predictive_result{
		csv::CsvDumper(path.insert(pos, "training_"), "", pair.second)};
	if (!predictive_result) {
		throw std::runtime_error(
			std::string{csv::CsvErrorMessage(predictive_result.error())});
	}
}
