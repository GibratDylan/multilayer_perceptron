#pragma once

#include "types/Types.hpp"

#include <string_view>
#include <vector>

namespace config_utils {

std::string_view Trim(std::string_view value);

bool ParseUnsigned(std::string_view token, ConfigCount* out);

bool ParseFloat(std::string_view token, LearningRate* out);

std::string_view StripInlineComment(std::string_view value);

std::vector<std::string> SplitTokens(std::string_view value);

bool ReportError(LineNumber line_number, std::string_view message);

}  // namespace config_utils
