#pragma once

#include "data/Dataset.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace csv {
bool GetSizeCSV(const std::string& path, int64_t& rows, int64_t& cols);
Dataset CsvLoader(const std::string& path);
std::pair<Dataset, Dataset> DatasetSplit(Dataset& dataset, float ratio);
void CsvDumper(const std::string& path, std::string_view header,
			   Dataset& dataset);

template <typename... Args>
Result<std::nullopt_t, CsvError> WriteToCsv(const std::string& path, std::string_view header,
				const Args&... args) {
	std::ofstream metrics_csv{path, std::ios::out | std::ios::app};

	if (!metrics_csv.good()) {
		return Result<std::nullopt_t, CsvError>{CsvError::kCannotOpen};
	}

	if (!metrics_csv.tellp() && !header.empty()) {
		metrics_csv << header << '\n';
	}

	bool first{true};
	auto write_arg = [&](const auto& arg) {
		if (!first) {
			metrics_csv << ',';
		}
		metrics_csv << arg;
		first = false;
	};

	(write_arg(args), ...);
	metrics_csv << '\n';
	return Result<std::nullopt_t, CsvError>{std::nullopt};
}
}  // namespace csv
