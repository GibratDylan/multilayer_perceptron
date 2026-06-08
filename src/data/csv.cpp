#include "data/csv.hpp"

#include "data/Dataset.hpp"
#include "types/eigen_types.hpp"

#include <cassert>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace csv {
std::optional<std::pair<int64_t, int64_t>> GetSizeCSV(const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		throw std::runtime_error("CSV can't be read in GetSizeCSV");
	}

	int64_t rows{};
	int64_t cols{};
	std::string line{};
	while (getline(file, line)) {
		++rows;

		std::string word{};
		std::stringstream line_stream{line};
		int64_t tmp_cols{};
		while (getline(line_stream, word, ',')) {
			if (rows == 1)
				++cols;
			else
				++tmp_cols;
		}

		if (rows != 1 && tmp_cols != cols) return std::nullopt;
	}

	if (!rows || !cols) return std::nullopt;

	return std::optional<std::pair<int64_t, int64_t>>{{rows, cols}};
}

Dataset CsvLoader(const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		throw std::runtime_error("CSV can't be read in CsvLoader");
	}

	Matrix dataset{};
	IntVector targe_data{};

	if (auto result = GetSizeCSV(path)) {
		auto [rows, cols] = *result;
		dataset = Matrix{cols - 1, rows};
		targe_data = IntVector{rows};
	} else {
		throw std::runtime_error("Size CSV not valid");
	}

	std::string line{};
	int64_t rows{};
	while (getline(file, line)) {
		int64_t cols{};
		bool seen{};
		std::string word{};
		std::stringstream line_stream{line};

		while (getline(line_stream, word, ',')) {
			if (cols != 1) {
				// We need to transpose data to row major, so we swap cols and
				// rows
				// VERIFICATION stof !!!!!!!!!!!!!!!
				// MODIFICATION COLS TARGET !!!!!!!!!!!!!!!
				// NOLINTBEGIN(readability-suspicious-call-argument)
				dataset(cols - seen, rows) = std::stof(word);
				// NOLINTEND(readability-suspicious-call-argument)
			} else {
				// Doit etre rendu plus clean
				seen = true;
				targe_data(rows) = word == "M";
			}

			++cols;
		}

		++rows;
	}

	return Dataset{std::move(dataset), std::move(targe_data)};
}

std::pair<Dataset, Dataset> DatasetSplit(Dataset& dataset, float ratio) {
	assert(ratio < 1 && ratio > 0);
	const int64_t size{
		static_cast<int64_t>(static_cast<float>(dataset.GetSize()) * ratio)};

	dataset.SetBatchSize(size);
	dataset.RandDataset();
	auto it = dataset.begin();

	Dataset first{(*it).first, (*it).second};
	++it;
	dataset.SetBatchSize(dataset.GetSize() - size);
	Dataset second{(*it).first, (*it).second};

	return {std::move(first), std::move(second)};
}

void CsvDumper(const std::string& path, std::string_view header,
			   Dataset& dataset) {
	dataset.SetBatchSize(1);

	for (auto&& [batch, target] : dataset) {
		std::string csv_line{};

		bool seen{};
		for (int64_t index{}; index < batch.size() + 1; ++index) {
			if (index) csv_line += ',';
			if (index == 1) {
				csv_line += target(0) ? 'M' : 'B';
				seen = true;
			} else {
				csv_line += std::to_string(batch(index - seen));
			}
		}

		WriteToCsv(path, header, csv_line);
	}
}
}  // namespace csv
