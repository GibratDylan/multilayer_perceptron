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
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

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

Result<Dataset, CsvError> CsvLoader(const std::string& path) {
	std::ifstream file{path};

	if (!file.is_open()) {
		return Result<Dataset, CsvError>{CsvError::kCannotOpen};
	}

	Matrix dataset{};
	IntVector target_data{};

	auto size_result{GetCsvSize(path)};
	if (size_result) {
		auto [rows, cols] = size_result.value();
		dataset = Matrix{cols - 1, rows};
		target_data = IntVector{rows};
	} else {
		return Result<Dataset, CsvError>{size_result.error()};
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

				if (result.ec == std::errc{} &&
					result.ptr == word.end().base()) {
					// NOLINTBEGIN(readability-suspicious-call-argument)
					dataset(cols, rows) = converted_value;
					// NOLINTEND(readability-suspicious-call-argument)
				} else {
					return Result<Dataset, CsvError>{CsvError::kBadFloat};
				}

				++cols;
			}
		}

		++rows;
	}

	return Result<Dataset, CsvError>{
		Dataset{std::move(dataset), std::move(target_data)}};
}

std::pair<Dataset, Dataset> DatasetSplit(Dataset& dataset, float ratio) {
	assert(ratio < 1 && ratio > 0);

	const int64_t size{static_cast<int64_t>(
		static_cast<float>(dataset.GetSizeCols()) * ratio)};

	const int64_t saved_batch{dataset.GetBatchSize()};
	dataset.SetBatchSize(size);
	dataset.RandDataset();
	auto it = dataset.begin();

	Dataset first{(*it).first, (*it).second};
	++it;
	dataset.SetBatchSize(dataset.GetSizeCols() - size);
	Dataset second{(*it).first, (*it).second};

	dataset.SetBatchSize(saved_batch);

	return {std::move(first), std::move(second)};
}

Result<std::monostate, CsvError> CsvDumper(const std::string& path,
										   std::string_view header,
										   Dataset& dataset) {
	const int64_t saved_batch{dataset.GetBatchSize()};
	dataset.SetBatchSize(1);

	auto writer_result{WriteToCsv::Create(path, header)};
	if (!writer_result) {
		dataset.SetBatchSize(saved_batch);
		return Result<std::monostate, CsvError>{writer_result.error()};
	}

	auto& csv_writer{writer_result.value()};

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
				if (result.ec == std::errc{}) {
					csv_line.append(buff.begin(), result.ptr - buff.begin());
				} else {
					dataset.SetBatchSize(saved_batch);
					return Result<std::monostate, CsvError>{
						CsvError::kCannotWrite};
				}
			}
		}

		auto write_result{csv_writer.Write(csv_line)};
		if (!write_result) {
			dataset.SetBatchSize(saved_batch);
			return Result<std::monostate, CsvError>{write_result.error()};
		}
	}

	dataset.SetBatchSize(saved_batch);

	return Result<std::monostate, CsvError>{std::monostate{}};
}

Result<WriteToCsv, CsvError> WriteToCsv::Create(const std::string& path,
												std::string_view header) {
	std::ofstream csv_file{path, std::ios::out | std::ios::trunc};
	if (!csv_file.is_open()) {
		return Result<WriteToCsv, CsvError>{CsvError::kCannotCreate};
	}

	if (!header.empty()) {
		csv_file << header << '\n';
		if (!csv_file) {
			return Result<WriteToCsv, CsvError>{CsvError::kCannotWrite};
		}
	}

	return Result<WriteToCsv, CsvError>{WriteToCsv{std::move(csv_file)}};
}

std::string_view CsvErrorMessage(CsvError error) noexcept {
	switch (error) {
		case CsvError::kCannotOpen:
			return "CSV can't be opened";
		case CsvError::kCannotCreate:
			return "CSV can't be created";
		case CsvError::kCannotWrite:
			return "CSV can't be written";
		case CsvError::kEmpty:
			return "CSV is empty";
		case CsvError::kNotRectangular:
			return "CSV is not rectangular";
		case CsvError::kBadFloat:
			return "CSV contains invalid numeric data";
	}

	return "CSV error";
}
}  // namespace csv
