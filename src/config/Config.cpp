#include "config/Config.hpp"

#include "activation/AActivation.hpp"
#include "config/configUtils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

Config::Config(std::string_view path)
	: size_{},
	  epochs_{},
	  batch_size_{},
	  learning_rate_{},
	  loss_func_{ALoss::NONE},
	  path_(path),
	  seen_epochs_{},
	  seen_learning_rate_{},
	  seen_batch_size_{},
	  seen_loss_{} {}

bool Config::parse() {
	std::ifstream file(path_);
	if (!file) {
		std::cerr << "Config parse error: cannot open file: " << path_ << '\n';
		return false;
	}
	reset();
	return parseLines(file);
}

void Config::reset() {
	size_ = 0;
	neuronal_layers_.clear();
	activation_func_.clear();
	epochs_ = 0;
	batch_size_ = 0;
	learning_rate_ = 0.0F;
	loss_func_ = ALoss::NONE;

	seen_epochs_ = false;
	seen_learning_rate_ = false;
	seen_batch_size_ = false;
	seen_loss_ = false;
}

bool Config::parseSingleValue(bool* seen, const Tokens& tokens,
							  size_t line_number, ParseSingle parse_and_set) {
	if (*seen) {
		return configUtils::ReportError(
			line_number, "duplicate '" + std::string(tokens[0]) + "' entry");
	}

	if (tokens.size() != 2) {
		return configUtils::ReportError(
			line_number, "invalid '" + std::string(tokens[0]) + "' format");
	}

	if (!(this->*parse_and_set)(tokens[1], line_number)) return false;

	*seen = true;
	return true;
}

bool Config::handleEpochs(const Tokens& tokens, size_t line_number) {
	return parseSingleValue(&seen_epochs_, tokens, line_number,
							&Config::parseEpochs);
}

bool Config::handleLearningRate(const Tokens& tokens, size_t line_number) {
	return parseSingleValue(&seen_learning_rate_, tokens, line_number,
							&Config::parseLearningRate);
}

bool Config::handleBatchSize(const Tokens& tokens, size_t line_number) {
	return parseSingleValue(&seen_batch_size_, tokens, line_number,
							&Config::parseBatchSize);
}

bool Config::handleLoss(const Tokens& tokens, size_t line_number) {
	return parseSingleValue(&seen_loss_, tokens, line_number,
							&Config::parseLoss);
}

bool Config::parseEpochs(std::string_view token, size_t line_number) {
	unsigned int value = 0;
	if (!configUtils::ParseUnsigned(token, &value) || value == 0) {
		return configUtils::ReportError(line_number, "invalid 'epochs' value");
	}
	epochs_ = value;
	return true;
}

bool Config::parseLearningRate(std::string_view token, size_t line_number) {
	float value = 0.0F;
	if (!configUtils::ParseFloat(token, &value) || value <= 0.0F) {
		return configUtils::ReportError(line_number,
										"invalid 'learning_rate' value");
	}
	learning_rate_ = value;
	return true;
}

bool Config::parseLoss(std::string_view token, size_t line_number) {
	if (!isLossFuncValid(token)) {
		return configUtils::ReportError(line_number, "invalid 'loss' value");
	}
	loss_func_ = ALoss::getLossType(token);
	return true;
}

bool Config::parseBatchSize(std::string_view token, size_t line_number) {
	unsigned int value = 0;
	if (!configUtils::ParseUnsigned(token, &value) || value == 0) {
		return configUtils::ReportError(line_number,
										"invalid 'batch_size' value");
	}
	batch_size_ = value;
	return true;
}

bool Config::parseLayer(const Tokens& tokens, size_t line_number) {
	if (tokens.size() != 4) {
		return configUtils::ReportError(line_number, "invalid 'layer' format");
	}
	unsigned int input_size = 0;
	unsigned int output_size = 0;
	if (!configUtils::ParseUnsigned(tokens[1], &input_size) ||
		input_size == 0 ||
		!configUtils::ParseUnsigned(tokens[2], &output_size) ||
		output_size == 0) {
		return configUtils::ReportError(line_number, "invalid 'layer' sizes");
	}
	if (!isActivationFuncValid(tokens[3])) {
		return configUtils::ReportError(line_number,
										"invalid activation function");
	}
	if (!neuronal_layers_.empty() && input_size != neuronal_layers_.back()) {
		return configUtils::ReportError(
			line_number, "layer input size does not match previous layer");
	}
	neuronal_layers_.push_back(output_size);
	activation_func_.push_back(AActivation::getActivationType(tokens[3]));
	size_++;
	return true;
}

Handlers Config::build_handlers() {
	return {
		{"epochs", &Config::handleEpochs},
		{"learning_rate", &Config::handleLearningRate},
		{"batch_size", &Config::handleBatchSize},
		{"loss", &Config::handleLoss},
		{"layer", &Config::parseLayer},
	};
}

bool Config::parseLines(std::ifstream& file) {
	std::string line;
	size_t line_number = 0;

	Handlers handlers = build_handlers();

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
		if (!(this->*it->second)(tokens, line_number)) return false;
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

size_t Config::getEpochs() const {
	return epochs_;
}

float Config::getLearningRate() const {
	return learning_rate_;
}

size_t Config::getBatchSize() const {
	return batch_size_;
}

const std::vector<size_t>& Config::getNeuralLayer() const {
	return neuronal_layers_;
}

const std::vector<AActivation::activationFuncType>& Config::getActivationFunc()
	const {
	return activation_func_;
}

ALoss::lossFuncType Config::getLossFunc() const {
	return loss_func_;
}

size_t Config::getSize() const {
	return size_;
}
