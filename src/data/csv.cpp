#include "data/csv.hpp"

#include "data/Dataset.hpp"

#include <cstdint>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace csv {
std::optional<std::pair<int64_t, int64_t>> GetSizeCSV(const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		throw std::runtime_error("CSV can't be read in CsvLoader");
	}

	int64_t rows{};
	int64_t cols{};
	std::string line{};
	while (getline(file, line)) {
		++rows;

		std::string word{};
		std::stringstream line_stream{line};
		int64_t tmp_cols{};
		while (getline(line_stream, word)) {
			if (rows == 1)
				++cols;
			else
				++tmp_cols;
		}

		if (tmp_cols != cols) return std::nullopt;
	}

	if (!rows || !cols) return std::nullopt;

	return std::optional<std::pair<int64_t, int64_t>>{{rows, cols}};
}

Dataset CsvLoader(const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		throw std::runtime_error("CSV can't be read in CsvLoader");
	}

	Matrix csv{};

	if (auto result = GetSizeCSV(path)) {
		auto [rows, cols] = *result;
		csv = Matrix{cols, rows};
	}

	std::string line{};
	while (getline(file, line)) {
		std::string word{};
		std::stringstream line_stream{line};

		while (getline(line_stream, word)) {
			// csv();
		}
	}

	// return {csv, }
}

std::pair<Dataset, Dataset> DatasetSplit(const Dataset& dataset,
										 int64_t ratio) {}

void CsvDumper(const Dataset& dataset) {}
}  // namespace csv
