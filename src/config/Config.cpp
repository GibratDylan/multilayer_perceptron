#include "config/Config.hpp"

#include "activation/AActivation.hpp"
#include "config/config_utils.hpp"
#include "loss/ALoss.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

Config::Config(std::string_view path)
	: path_(path) {}

bool Config::Parse() {
	assert(!path_.empty());
	std::ifstream file{path_};
	if (!file) {
		std::cerr << "Config parse error: cannot open file: " << path_ << '\n';
		return false;
	}
	Reset();
	return ParseLines(file);
}

void Config::Reset() {
	size_ = 0;
	input_size_ = 0;
	neuronal_layers_.clear();
	activation_func_.clear();
	epochs_ = 0;
	batch_size_ = 0;
	learning_rate_ = 0.0F;
	loss_func_ = {};

	seen_epochs_ = false;
	seen_learning_rate_ = false;
	seen_batch_size_ = false;
	seen_input_size_ = false;
	seen_loss_func_ = false;
}

bool Config::ParseSingleValue(bool* seen, const Tokens& tokens,
							  int64_t line_number, ParseSingle parse_and_set) {
	assert(seen != nullptr);
	assert(parse_and_set != nullptr);
	assert(!tokens.empty());
	if (*seen) {
		return config_utils::ReportError(
			line_number, "duplicate '" + std::string(tokens.at(0)) + "' entry");
	}

	if (tokens.size() != 2) {
		return config_utils::ReportError(
			line_number, "invalid '" + std::string(tokens.at(0)) + "' format");
	}

	if (!(this->*parse_and_set)(tokens.at(1), line_number)) return false;

	*seen = true;
	return true;
}

bool Config::HandleEpochs(const Tokens& tokens, int64_t line_number) {
	return ParseSingleValue(&seen_epochs_, tokens, line_number,
							&Config::ParseEpochs);
}

bool Config::HandleLearningRate(const Tokens& tokens, int64_t line_number) {
	return ParseSingleValue(&seen_learning_rate_, tokens, line_number,
							&Config::ParseLearningRate);
}

bool Config::HandleBatchSize(const Tokens& tokens, int64_t line_number) {
	return ParseSingleValue(&seen_batch_size_, tokens, line_number,
							&Config::ParseBatchSize);
}

bool Config::HandleInputSize(const Tokens& tokens, int64_t line_number) {
	return ParseSingleValue(&seen_input_size_, tokens, line_number,
							&Config::ParseInputSize);
}

bool Config::HandleLossFunc(const Tokens& tokens, int64_t line_number) {
	return ParseSingleValue(&seen_loss_func_, tokens, line_number,
							&Config::ParseLossFunc);
}

bool Config::ParseEpochs(std::string_view token, int64_t line_number) {
	int64_t value{};
	if (!config_utils::ParseSigned(token, &value) || value <= 0) {
		return config_utils::ReportError(line_number, "invalid 'epochs' value");
	}
	epochs_ = value;
	return true;
}

bool Config::ParseLearningRate(std::string_view token, int64_t line_number) {
	float value{};
	if (!config_utils::ParseFloat(token, &value) || value <= 0.0F) {
		return config_utils::ReportError(line_number,
										 "invalid 'learning_rate' value");
	}
	learning_rate_ = value;
	return true;
}

bool Config::ParseInputSize(std::string_view token, int64_t line_number) {
	int64_t value{};
	if (!config_utils::ParseSigned(token, &value) || value <= 0) {
		return config_utils::ReportError(line_number,
										 "invalid 'input_size' value");
	}
	input_size_ = value;
	neuronal_layers_.insert(neuronal_layers_.begin(), value);
	size_++;
	return true;
}

bool Config::ParseLossFunc(std::string_view token, int64_t line_number) {
	auto result = ALoss::GetLossType(token);
	if (!result)
		return config_utils::ReportError(line_number, "invalid 'loss' value");
	loss_func_ = result.value();
	return true;
}

bool Config::ParseBatchSize(std::string_view token, int64_t line_number) {
	int64_t value{};
	if (!config_utils::ParseSigned(token, &value) || value <= 0) {
		return config_utils::ReportError(line_number,
										 "invalid 'batch_size' value");
	}
	batch_size_ = value;
	return true;
}

bool Config::ParseLayer(const Tokens& tokens, int64_t line_number) {
	if (tokens.size() != 3) {
		return config_utils::ReportError(line_number, "invalid 'layer' format");
	}
	int64_t output_size{};
	if (!config_utils::ParseSigned(tokens.at(1), &output_size) ||
		output_size <= 0) {
		return config_utils::ReportError(line_number, "invalid 'layer' sizes");
	}
	auto result = AActivation::GetActivationType(tokens.at(2));
	if (!result)
		return config_utils::ReportError(line_number,
										 "invalid activation function");
	neuronal_layers_.push_back(output_size);
	activation_func_.push_back(result.value());
	size_++;
	return true;
}

Config::Handlers Config::BuildHandlers() {
	return {
		{"epochs", &Config::HandleEpochs},
		{"learning_rate", &Config::HandleLearningRate},
		{"batch_size", &Config::HandleBatchSize},
		{"input_size", &Config::HandleInputSize},
		{"loss", &Config::HandleLossFunc},
		{"layer", &Config::ParseLayer},
	};
}

bool Config::ParseLines(std::ifstream& file) {
	std::string line{};
	int64_t line_number{};

	Handlers handlers{BuildHandlers()};
	assert(!handlers.empty());

	while (std::getline(file, line)) {
		++line_number;
		std::string_view view{config_utils::Trim(
			config_utils::StripInlineComment(config_utils::Trim(line)))};
		if (view.empty()) continue;

		Tokens tokens{config_utils::SplitTokens(view)};
		if (tokens.empty()) continue;
		const std::string& key{tokens.front()};
		assert(!key.empty());
		const auto it{handlers.find(key)};
		if (it == handlers.end()) {
			return config_utils::ReportError(line_number,
											 "unknown key '" + key + "'");
		}
		if (!(this->*it->second)(tokens, line_number)) return false;
	}

	if (!IsConfigValid()) {
		std::cerr
			<< "Config parse error: incomplete or invalid configuration\n";
		return false;
	}

	return true;
}

bool Config::IsConfigValid() const noexcept {
	if (epochs_ <= 0 || !seen_epochs_) return false;
	if (batch_size_ <= 0 || !seen_batch_size_) return false;
	if (input_size_ <= 0 || !seen_input_size_) return false;
	if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0F ||
		!seen_learning_rate_)
		return false;
	if (!seen_loss_func_) return false;
	if (neuronal_layers_.empty()) return false;
	if (size_ <= 0) return false;
	if (activation_func_.size() != static_cast<size_t>(size_ - 1)) return false;
	if (neuronal_layers_.size() != static_cast<size_t>(size_)) return false;
	if (neuronal_layers_.front() != input_size_) return false;
	if (std::any_of(neuronal_layers_.cbegin(), neuronal_layers_.cend(),
					[](int64_t layer_size) { return layer_size == 0; }))
		return false;
	return true;
}

int64_t Config::GetEpochs() const noexcept {
	return epochs_;
}

float Config::GetLearningRate() const noexcept {
	return learning_rate_;
}

int64_t Config::GetBatchSize() const noexcept {
	return batch_size_;
}

int64_t Config::GetInputSize() const noexcept {
	return input_size_;
}

const std::vector<int64_t>& Config::GetNeuralLayer() const noexcept {
	return neuronal_layers_;
}

const std::vector<AActivation::ActivationFuncType>& Config::GetActivationFunc()
	const noexcept {
	return activation_func_;
}

ALoss::LossFuncType Config::GetLossFunc() const noexcept {
	return loss_func_;
}

int64_t Config::GetSize() const noexcept {
	return size_;
}
