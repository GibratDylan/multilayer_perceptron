#include "config/Config.hpp"

#include "activation/AActivation.hpp"
#include "config/config_utils.hpp"

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
	  loss_func_{ALoss::kNone},
	  path_(path),
	  seen_epochs_{},
	  seen_learning_rate_{},
	  seen_batch_size_{},
	  seen_loss_{} {}

bool Config::Parse() {
	std::ifstream file(path_);
	if (!file) {
		std::cerr << "Config parse error: cannot open file: " << path_ << '\n';
		return false;
	}
	Reset();
	return ParseLines(file);
}

void Config::Reset() {
	size_ = 0;
	neuronal_layers_.clear();
	activation_func_.clear();
	epochs_ = 0;
	batch_size_ = 0;
	learning_rate_ = 0.0;
	loss_func_ = ALoss::kNone;

	seen_epochs_ = false;
	seen_learning_rate_ = false;
	seen_batch_size_ = false;
	seen_loss_ = false;
}

bool Config::ParseSingleValue(bool* seen, const Tokens& tokens,
							  int64_t line_number, ParseSingle parse_and_set) {
	if (*seen) {
		return config_utils::ReportError(
			line_number, "duplicate '" + std::string(tokens[0]) + "' entry");
	}

	if (tokens.size() != 2) {
		return config_utils::ReportError(
			line_number, "invalid '" + std::string(tokens[0]) + "' format");
	}

	if (!(this->*parse_and_set)(tokens[1], line_number)) return false;

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

bool Config::HandleLoss(const Tokens& tokens, int64_t line_number) {
	return ParseSingleValue(&seen_loss_, tokens, line_number,
							&Config::ParseLoss);
}

bool Config::ParseEpochs(std::string_view token, int64_t line_number) {
	int64_t value = 0;
	if (!config_utils::ParseSigned(token, &value) || value == 0) {
		return config_utils::ReportError(line_number, "invalid 'epochs' value");
	}
	epochs_ = value;
	return true;
}

bool Config::ParseLearningRate(std::string_view token, int64_t line_number) {
	float value = 0.0;
	if (!config_utils::ParseFloat(token, &value) || value <= 0.0) {
		return config_utils::ReportError(line_number,
										 "invalid 'learning_rate' value");
	}
	learning_rate_ = value;
	return true;
}

bool Config::ParseLoss(std::string_view token, int64_t line_number) {
	if (!IsLossFuncValid(token)) {
		return config_utils::ReportError(line_number, "invalid 'loss' value");
	}
	loss_func_ = ALoss::GetLossType(token);
	return true;
}

bool Config::ParseBatchSize(std::string_view token, int64_t line_number) {
	int64_t value = 0;
	if (!config_utils::ParseSigned(token, &value) || value == 0) {
		return config_utils::ReportError(line_number,
										 "invalid 'batch_size' value");
	}
	batch_size_ = value;
	return true;
}

bool Config::ParseLayer(const Tokens& tokens, int64_t line_number) {
	if (tokens.size() != 4) {
		return config_utils::ReportError(line_number, "invalid 'layer' format");
	}
	int64_t input_size = 0;
	int64_t output_size = 0;
	if (!config_utils::ParseSigned(tokens[1], &input_size) || input_size == 0 ||
		!config_utils::ParseSigned(tokens[2], &output_size) ||
		output_size == 0) {
		return config_utils::ReportError(line_number, "invalid 'layer' sizes");
	}
	if (!IsActivationFuncValid(tokens[3])) {
		return config_utils::ReportError(line_number,
										 "invalid activation function");
	}
	if (!neuronal_layers_.empty() && input_size != neuronal_layers_.back()) {
		return config_utils::ReportError(
			line_number, "layer input size does not match previous layer");
	}
	neuronal_layers_.push_back(output_size);
	activation_func_.push_back(AActivation::GetActivationType(tokens[3]));
	size_++;
	return true;
}

Config::Handlers Config::BuildHandlers() {
	return {
		{"epochs", &Config::HandleEpochs},
		{"learning_rate", &Config::HandleLearningRate},
		{"batch_size", &Config::HandleBatchSize},
		{"loss", &Config::HandleLoss},
		{"layer", &Config::ParseLayer},
	};
}

bool Config::ParseLines(std::ifstream& file) {
	std::string line;
	int64_t line_number = 0;

	Handlers handlers = BuildHandlers();

	while (std::getline(file, line)) {
		++line_number;
		std::string_view view = config_utils::Trim(
			config_utils::StripInlineComment(config_utils::Trim(line)));
		if (view.empty()) continue;

		Tokens tokens = config_utils::SplitTokens(view);
		if (tokens.empty()) continue;
		const std::string& key = tokens.front();
		const auto it = handlers.find(key);
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

bool Config::IsLossFuncValid(std::string_view string) {
	return ALoss::GetLossType(string) != ALoss::kNone;
}

bool Config::IsActivationFuncValid(std::string_view string) {
	return AActivation::GetActivationType(string) != AActivation::kNone;
}

bool Config::IsConfigValid() const {
	if (epochs_ == 0 || batch_size_ == 0) return false;
	if (!std::isfinite(learning_rate_) || learning_rate_ <= 0.0) return false;
	if (loss_func_ == ALoss::kNone) return false;
	if (neuronal_layers_.empty()) return false;
	if (activation_func_.size() != neuronal_layers_.size()) return false;
	if (size_ == 0) return false;
	if (std::any_of(neuronal_layers_.cbegin(), neuronal_layers_.cend(),
					[](int64_t layer_size) { return layer_size == 0; }))
		return false;
	return true;
}

int64_t Config::GetEpochs() const {
	return epochs_;
}

float Config::GetLearningRate() const {
	return learning_rate_;
}

int64_t Config::GetBatchSize() const {
	return batch_size_;
}

const std::vector<int64_t>& Config::GetNeuralLayer() const {
	return neuronal_layers_;
}

const std::vector<AActivation::ActivationFuncType>& Config::GetActivationFunc()
	const {
	return activation_func_;
}

ALoss::LossFuncType Config::GetLossFunc() const {
	return loss_func_;
}

int64_t Config::GetSize() const {
	return size_;
}
