#pragma once

#include "data/Dataset.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string_view>

namespace csv {
bool GetSizeCSV(const std::string& path, int64_t& rows, int64_t& cols);
Dataset CsvLoader(std::string_view path);
std::pair<Dataset, Dataset> DatasetSplit(const Dataset& dataset, int64_t ratio);
void CsvDumper(const Dataset& dataset);

template <typename... Args>
void WriteToCsv(const std::string& path, std::string_view header,
				Args... args) {
	std::ofstream metrics_csv{path, std::ios::out | std::ios::app};

	if (!metrics_csv.good()) {
		std::cerr << "std::ofstream fail !\n";
		return;
	}

	if (!metrics_csv.tellp()) {
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
}
}  // namespace csv
