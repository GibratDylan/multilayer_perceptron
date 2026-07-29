#include "data/csv.hpp"

#include "data/Dataset.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"
#include "types/eigen_types.hpp"

#include <array>
#include <cassert>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace csv {
Result<std::pair<int64_t, int64_t>, CsvError> GetCsvSize(
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
	IntVector target_data{};

	if (auto result = GetCsvSize(path)) {
		auto [rows, cols] = result.value();
		dataset = Matrix{cols - 1, rows};
		target_data = IntVector{rows};
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
		std::string word{};
		std::stringstream line_stream{line};

		while (getline(line_stream, word, ',')) {
			if (word == "M" || word == "B") {
				target_data(rows) = word == "M";
			} else {
				float converted_value{};
				auto result{std::from_chars(
					word.begin().base(), word.end().base(), converted_value)};

				if (result.ec == std::errc{}) {
					if (result.ptr != word.end().base()) {
						throw std::runtime_error("CSV data not valid");
						// NOLINTBEGIN(readability-suspicious-call-argument)
						dataset(cols, rows) = converted_value;
						// NOLINTEND(readability-suspicious-call-argument)
					}
				} else {
					throw std::runtime_error(
						std::make_error_code(result.ec).message());
				}

				++cols;
			}
		}

		++rows;
	}

	return Dataset{std::move(dataset), std::move(target_data)};
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
	const int64_t saved_batch = dataset.GetBatchSize();
	dataset.SetBatchSize(1);

	WriteToCsv csv_writer{path, header};

	for (auto&& [batch, target] : dataset) {
		std::string csv_line{};
		int64_t cols{};

		for (int64_t index{}; index < batch.size() + 1; ++index) {
			if (index) csv_line += ',';
			if (index == 1) {
				csv_line += target(0) ? 'M' : 'B';
			} else {
				const std::size_t buff_size = 10;
				std::array<char, buff_size> buff{};
				auto result{
					std::to_chars(buff.begin(), buff.end(), batch(cols))};
				++cols;
				if (result.ec == std::errc{})
					csv_line.append(buff.begin(), result.ptr);
				else
					throw std::runtime_error(
						std::make_error_code(result.ec).message());
			}
		}

		csv_writer.Write(csv_line);
	}

	dataset.SetBatchSize(saved_batch);
}
}  // namespace csv
