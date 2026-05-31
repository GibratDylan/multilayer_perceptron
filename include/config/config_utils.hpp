#pragma once

#include <string_view>
#include <vector>
#include <cstdint>

namespace config_utils {

std::string_view Trim(std::string_view value);

bool ParseSigned(std::string_view token, int64_t* out);

bool ParseFloat(std::string_view token, float* out);

std::string_view StripInlineComment(std::string_view value);

std::vector<std::string> SplitTokens(std::string_view value);

bool ReportError(int64_t line_number, std::string_view message);

}  // namespace config_utils
