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
		std::string word{};
		std::stringstream line_stream{line};

		while (getline(line_stream, word, ',')) {
			if (cols != 1) {
				// We need to transpose data to row major, so we swap cols and
				// rows
				// VERIFICATION stof !!!!!!!!!!!!!!!
				// MODIFICATION COLS TARGET !!!!!!!!!!!!!!!
				// NOLINTBEGIN(readability-suspicious-call-argument)
				dataset(cols, rows) = std::stof(word);
				// NOLINTEND(readability-suspicious-call-argument)
			} else {
				// Doit etre rendu plus clean
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
	auto it = dataset.begin();

	Dataset first{(*it).first, (*it).second};
	++it;
	Dataset second{(*it).first, (*it).second};

	return {std::move(first), std::move(second)};
}

void CsvDumper(const std::string& path, std::string_view header,
			   Dataset& dataset) {
	dataset.SetBatchSize(1);

	for (auto&& [batch, target] : dataset) {
		std::string batch_line{};

		for (int64_t index{}; index < batch.size(); ++index) {
			if (index) batch_line += ',';
			batch_line += std::to_string(batch(index));
		}

		WriteToCsv(path, header, batch_line, target(0));
	}
}
}  // namespace csv
