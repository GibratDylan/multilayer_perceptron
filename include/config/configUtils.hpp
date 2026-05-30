#pragma once

#include <string_view>
#include <vector>

namespace configUtils {

std::string_view Trim(std::string_view value);

bool ParseUnsigned(std::string_view token, unsigned int* out);

bool ParseFloat(std::string_view token, float* out);

std::string_view StripInlineComment(std::string_view value);

std::vector<std::string> SplitTokens(std::string_view value);

bool ReportError(size_t line_number, std::string_view message);

}  // namespace configUtils
