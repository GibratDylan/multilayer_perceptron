#pragma once

#include "data/Dataset.hpp"

#include <cstdint>
#include <string_view>

namespace csv_dataset {
const Dataset&& CsvLoader(std::string_view path);
const std::pair<Dataset, Dataset>&& DatasetSplit(const Dataset& dataset,
												 int64_t ratio);
void CsvDumper(const Dataset& dataset);
}  // namespace csv_dataset
