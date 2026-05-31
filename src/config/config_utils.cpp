#include "config/config_utils.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace config_utils {
std::string_view Trim(std::string_view value) {
	while (!value.empty() &&
		   std::isspace(static_cast<unsigned char>(value.front())) != 0) {
		value.remove_prefix(1);
	}
	while (!value.empty() &&
		   std::isspace(static_cast<unsigned char>(value.back())) != 0) {
		value.remove_suffix(1);
	}
	return value;
}

bool ParseSigned(std::string_view token, int64_t* out) {
	if (token.empty() || out == nullptr) return false;
	std::size_t pos = 0;
	int64_t value = 0;
	try {
		value = std::stoi(std::string(token), &pos);
	} catch (const std::invalid_argument&) {
		return false;
	} catch (const std::out_of_range&) {
		return false;
	}
	if (pos != token.size()) return false;
	if (value > std::numeric_limits<int>::max()) return false;
	*out = value;
	return true;
}

bool ParseFloat(std::string_view token, float* out) {
	if (token.empty() || out == nullptr) return false;
	std::size_t pos = 0;
	float value = 0.F;
	try {
		value = std::stof(std::string(token), &pos);
	} catch (const std::invalid_argument&) {
		return false;
	} catch (const std::out_of_range&) {
		return false;
	}
	if (pos != token.size()) return false;
	if (!std::isfinite(value)) return false;
	*out = value;
	return true;
}

std::string_view StripInlineComment(std::string_view value) {
	std::size_t hash_pos = value.find('#');
	std::size_t slash_pos = value.find("//");
	std::size_t cut_pos = std::string_view::npos;
	if (hash_pos != std::string_view::npos) cut_pos = hash_pos;
	if (slash_pos != std::string_view::npos) {
		cut_pos = (cut_pos == std::string_view::npos)
					  ? slash_pos
					  : std::min(cut_pos, slash_pos);
	}
	if (cut_pos == std::string_view::npos) return value;
	return value.substr(0, cut_pos);
}

std::vector<std::string> SplitTokens(std::string_view value) {
	std::istringstream iss(std::string{value});
	std::vector<std::string> tokens;
	std::string token;
	while (iss >> token) tokens.push_back(token);
	return tokens;
}

bool ReportError(int64_t line_number, std::string_view message) {
	std::cerr << "Config parse error at line " << line_number << ": " << message
			  << '\n';
	return false;
}
}  // namespace config_utils
