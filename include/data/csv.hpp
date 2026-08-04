#pragma once

#include "data/Dataset.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace csv {
Result<std::pair<int64_t, int64_t>, CsvError> GetCsvSize(
	const std::string& path);
Result<Dataset, CsvError> CsvLoader(const std::string& path);
std::pair<Dataset, Dataset> DatasetSplit(Dataset& dataset, float ratio);
Result<std::monostate, CsvError> CsvDumper(const std::string& path,
										   std::string_view header,
										   Dataset& dataset);

std::string_view CsvErrorMessage(CsvError error) noexcept;

class WriteToCsv {
   private:
	std::ofstream csv_file_{};

	explicit WriteToCsv(std::ofstream&& csv_file)
		: csv_file_{std::move(csv_file)} {}

   public:
	explicit WriteToCsv() = delete;

	static Result<WriteToCsv, CsvError> Create(const std::string& path,
											   std::string_view header);

	template <typename... Args>
	Result<std::monostate, CsvError> Write(const Args&... args) {
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
		if (!csv_file_)
			return Result<std::monostate, CsvError>{CsvError::kCannotWrite};

		return Result<std::monostate, CsvError>{std::monostate{}};
	}
};
}  // namespace csv
