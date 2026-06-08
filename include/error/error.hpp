#pragma once

#include <cstdint>

enum class CsvError : uint8_t {
  kCannotOpen,
  kEmpty,
  kNotRectangular,
  kBadFloat,
};