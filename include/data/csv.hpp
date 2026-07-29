#pragma once

#include "data/Dataset.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace csv {
bool GetCsvSize(const std::string& path, int64_t& rows, int64_t& cols);
Dataset CsvLoader(const std::string& path);
std::pair<Dataset, Dataset> DatasetSplit(Dataset& dataset, float ratio);
void CsvDumper(const std::string& path, std::string_view header,
			   Dataset& dataset);

class WriteToCsv {
   private:
	std::ofstream csv_file_{};

   public:
	explicit WriteToCsv(const std::string& path, std::string_view header)
		: csv_file_{path, std::ios::out | std::ios::trunc} {
		
			if (!csv_file_.is_open()) {
			throw std::runtime_error("CSV at " + path + " can't be create");
		}

		csv_file_ << header << '\n';
	}

	template <typename... Args>
	void Write(const Args&... args) {
		bool first{true};

		auto write_arg = [&](const auto& arg) {
			if (!first) {
				csv_file_ << ',';
			}
			csv_file_ << arg;
			first = false;
		};

		(write_arg(args), ...);
		
		csv_file_ << '\n';
	}
};
}  // namespace csv
