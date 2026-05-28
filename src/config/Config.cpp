#include "config/Config.hpp"

#include "activation/AActivation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace configUtils {

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

bool ParseUnsigned(std::string_view token, unsigned int* out) {
	if (token.empty() || out == nullptr) return false;
	std::size_t pos = 0;
	unsigned long value = 0;
	try {
		value = std::stoul(std::string(token), &pos);
	} catch (const std::invalid_argument&) {
		return false;
	} catch (const std::out_of_range&) {
		return false;
	}
	if (pos != token.size()) return false;
	if (value > std::numeric_limits<unsigned int>::max()) return false;
	*out = static_cast<unsigned int>(value);
	return true;
}

bool ParseFloat(std::string_view token, float* out) {
	if (token.empty() || out == nullptr) return false;
	std::size_t pos = 0;
	float value = 0.0F;
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

bool ReportError(std::size_t line_number, std::string_view message) {
	std::cerr << "Config parse error at line " << line_number << ": " << message
			  << '\n';
	return false;
}

}  // namespace configUtils

Config::Config(std::string_view path)
	: size_{0},
	  epochs_{0},
	  batch_size_{0},
	  learning_rate_{0},
	  loss_func_{ALoss::NONE},
	  path_(path) {}

bool Config::parse() {
	std::ifstream file(path_);
	if (!file) {
		std::cerr << "Config parse error: cannot open file: " << path_ << '\n';
		return false;
	}

	size_ = 0;
	neuronal_layers_.clear();
	activation_func_.clear();
	epochs_ = 0;
	batch_size_ = 0;
	learning_rate_ = 0.0F;
	loss_func_ = ALoss::NONE;

	bool seen_epochs = false;
	bool seen_learning_rate = false;
	bool seen_batch_size = false;
	bool seen_loss = false;

	std::string line;
	std::size_t line_number = 0;

	using Tokens = std::vector<std::string>;

	std::unordered_map<std::string, std::function<bool(const Tokens&)>>
		handlers;

	auto parse_single_value =
		[&](bool* seen, std::string_view name, const Tokens& tokens,
			const std::function<bool(std::string_view)>& parse_and_set)
		-> bool {
		if (*seen) {
			return configUtils::ReportError(
				line_number, "duplicate '" + std::string(name) + "' entry");
		}
		if (tokens.size() != 2) {
			return configUtils::ReportError(
				line_number, "invalid '" + std::string(name) + "' format");
		}
		if (!parse_and_set(tokens[1])) return false;
		*seen = true;
		return true;
	};

	handlers.emplace("epochs", [&](const Tokens& tokens) {
		return parse_single_value(
			&seen_epochs, "epochs", tokens, [&](std::string_view token) {
				unsigned int value = 0;
				if (!configUtils::ParseUnsigned(token, &value) || value == 0) {
					return configUtils::ReportError(line_number,
													"invalid 'epochs' value");
				}
				epochs_ = value;
				return true;
			});
	});

	handlers.emplace("learning_rate", [&](const Tokens& tokens) {
		return parse_single_value(
			&seen_learning_rate, "learning_rate", tokens,
			[&](std::string_view token) {
				float value = 0.0F;
				if (!configUtils::ParseFloat(token, &value) || value <= 0.0F) {
					return configUtils::ReportError(
						line_number, "invalid 'learning_rate' value");
				}
				learning_rate_ = value;
				return true;
			});
	});

	handlers.emplace("batch_size", [&](const Tokens& tokens) {
		return parse_single_value(
			&seen_batch_size, "batch_size", tokens,
			[&](std::string_view token) {
				unsigned int value = 0;
				if (!configUtils::ParseUnsigned(token, &value) || value == 0) {
					return configUtils::ReportError(
						line_number, "invalid 'batch_size' value");
				}
				batch_size_ = value;
				return true;
			});
	});

	handlers.emplace("loss", [&](const Tokens& tokens) {
		return parse_single_value(
			&seen_loss, "loss", tokens, [&](std::string_view token) {
				if (!isLossFuncValid(token)) {
					return configUtils::ReportError(line_number,
													"invalid 'loss' value");
				}
				loss_func_ = ALoss::getLossType(token);
				return true;
			});
	});

	handlers.emplace("layer", [&](const Tokens& tokens) {
		if (tokens.size() != 4) {
			return configUtils::ReportError(line_number,
											"invalid 'layer' format");
		}
		unsigned int input_size = 0;
		unsigned int output_size = 0;
		if (!configUtils::ParseUnsigned(tokens[1], &input_size) ||
			input_size == 0 ||
			!configUtils::ParseUnsigned(tokens[2], &output_size) ||
			output_size == 0) {
			return configUtils::ReportError(line_number,
											"invalid 'layer' sizes");
		}
		if (!isActivationFuncValid(tokens[3])) {
			return configUtils::ReportError(line_number,
											"invalid activation function");
		}
		if (!neuronal_layers_.empty() &&
			input_size != neuronal_layers_.back()) {
			return configUtils::ReportError(
				line_number, "layer input size does not match previous layer");
		}
		neuronal_layers_.push_back(output_size);
		activation_func_.push_back(AActivation::getActivationType(tokens[3]));
		size_++;
		return true;
	});

	while (std::getline(file, line)) {
		++line_number;
		std::string_view view = configUtils::Trim(
			configUtils::StripInlineComment(configUtils::Trim(line)));
		if (view.empty()) continue;

		Tokens tokens = configUtils::SplitTokens(view);
		if (tokens.empty()) continue;
		const std::string& key = tokens.front();
		const auto it = handlers.find(key);
		if (it == handlers.end()) {
			return configUtils::ReportError(line_number,
											"unknown key '" + key + "'");
		}
		if (!it->second(tokens)) return false;
	}

	if (!isConfigValid()) {
		std::cerr
			<< "Config parse error: incomplete or invalid configuration\n";
		return false;
	}

	return true;
}

bool Config::isLossFuncValid(std::string_view string) {
	return ALoss::getLossType(string) != ALoss::NONE;
}

bool Config::isActivationFuncValid(std::string_view string) {
	return AActivation::getActivationType(string) != AActivation::NONE;
}

bool Config::isConfigValid() const {
	if (epochs_ == 0 || batch_size_ == 0) return false;
	if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F) return false;
	if (loss_func_ == ALoss::NONE) return false;
	if (neuronal_layers_.empty()) return false;
	if (activation_func_.size() != neuronal_layers_.size()) return false;
	if (size_ == 0) return false;
	if (std::any_of(neuronal_layers_.cbegin(), neuronal_layers_.cend(),
					[](unsigned int layer_size) { return layer_size == 0; }))
		return false;
	return true;
}

unsigned int Config::getEpochs() const {
	return epochs_;
}

float Config::getLearningRate() const {
	return learning_rate_;
}

unsigned int Config::getBatchSize() const {
	return batch_size_;
}

const std::vector<unsigned int>& Config::getNeuralLayer() const {
	return neuronal_layers_;
}

const std::vector<AActivation::activationFuncType>& Config::getActivationFunc()
	const {
	return activation_func_;
}

ALoss::lossFuncType Config::getLossFunc() const {
	return loss_func_;
}

unsigned int Config::getSize() const {
	return size_;
}
