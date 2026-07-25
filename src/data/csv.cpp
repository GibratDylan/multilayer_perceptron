#include "data/csv.hpp"

#include "data/Dataset.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"
#include "types/eigen_types.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace csv {
Result<std::pair<int64_t, int64_t>, CsvError> GetSizeCSV(
	const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		return Result<std::pair<int64_t, int64_t>, CsvError>{
			CsvError::kCannotOpen};
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

		if (rows != 1 && tmp_cols != cols)
			return Result<std::pair<int64_t, int64_t>, CsvError>{
				CsvError::kNotRectangular};
	}

	if (!rows || !cols)
		return Result<std::pair<int64_t, int64_t>, CsvError>{CsvError::kEmpty};

	return Result<std::pair<int64_t, int64_t>, CsvError>{{rows, cols}};
}

Dataset CsvLoader(const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		throw std::runtime_error("CSV can't be read in CsvLoader");
	}

	Matrix dataset{};
	IntVector targe_data{};

	if (auto result = GetSizeCSV(path)) {
		auto [rows, cols] = result.value();
		dataset = Matrix{cols - 1, rows};
		targe_data = IntVector{rows};
	} else if (result.error() == CsvError::kEmpty) {
		throw std::runtime_error("Size CSV not valid");
	} else if (result.error() == CsvError::kNotRectangular) {
		throw std::runtime_error("CSV not Rectangular");
	} else if (result.error() == CsvError::kCannotOpen) {
		throw std::runtime_error("CSV can't be open");
	}

	std::string line{};
	int64_t rows{};
	while (getline(file, line)) {
		int64_t cols{};
		bool seen_expected_result{};
		std::string word{};
		std::stringstream line_stream{line};

		while (getline(line_stream, word, ',')) {
			std::size_t pos{};
			if (word == "M" || word == "B") {
				seen_expected_result = true;
				///////////// MODIFICATION COLS TARGET !!!!!!!!!!!!!!!
				////////////////////
				targe_data(rows) = word == "M";
			} else {
				float converted_value{};
				try {
					converted_value = std::stof(word, &pos);
				} catch (std::invalid_argument& e) {
					throw std::runtime_error("Out of range in CSV");
				} catch (std::out_of_range& e) {
					throw std::runtime_error(
						"CSV expected value to predict not valid");
				}
				if (word[pos] != '\0')
					throw std::runtime_error("CSV data not valid");
				// NOLINTBEGIN(readability-suspicious-call-argument)
				dataset(cols - seen_expected_result, rows) = converted_value;
				// NOLINTEND(readability-suspicious-call-argument)

				++cols;
			}

			++rows;
		}
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

		bool seen_expected_result{};
		for (int64_t index{}; index < batch.size() + 1; ++index) {
			if (index) csv_line += ',';
			if (index == 1) {
				csv_line += target(0) ? 'M' : 'B';
				seen_expected_result = true;
			} else {
				csv_line += std::to_string(batch(index - seen_expected_result));
			}
		}

		auto result = WriteToCsv(path, header, csv_line);
		if (result.error() == CsvError::kCannotOpen) {
			throw std::runtime_error("CSV at " + path + " can't be create");
		}
	}
}
}  // namespace csv
