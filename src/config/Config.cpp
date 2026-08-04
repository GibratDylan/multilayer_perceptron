#include "config/Config.hpp"

#include "activation/AActivation.hpp"
#include "config/config_utils.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"
#include "loss/ALoss.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {
Result<std::monostate, ConfigError> MakeConfigOk() {
	return Result<std::monostate, ConfigError>{std::monostate{}};
}

Result<std::monostate, ConfigError> MakeConfigError(ConfigError::Code code,
													int64_t line_number) {
	return Result<std::monostate, ConfigError>{ConfigError{code, line_number}};
}

Result<std::monostate, ConfigError> MakeConfigError(ConfigError::Code code) {
	return MakeConfigError(code, 0);
}
}  // namespace

Config::Config(std::string_view path) : path_(path) {}

Config::ParseResult Config::Parse() {
	assert(!path_.empty());
	std::ifstream file{path_};
	if (!file) {
		return MakeConfigError(ConfigError::Code::kCannotOpen);
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
	learning_rate_ = 0.0F;
	loss_func_ = {};

	seen_epochs_ = false;
	seen_learning_rate_ = false;
	seen_batch_size_ = false;
	seen_loss_func_ = false;
}

Config::ParseResult Config::ParseSingleValue(bool* seen, const Tokens& tokens,
											 int64_t line_number,
											 ParseSingle parse_and_set) {
	assert(seen != nullptr);
	assert(parse_and_set != nullptr);
	assert(!tokens.empty());
	if (*seen) {
		return MakeConfigError(ConfigError::Code::kDuplicateEntry, line_number);
	}

	if (tokens.size() != 2) {
		return MakeConfigError(ConfigError::Code::kInvalidFormat, line_number);
	}

	auto result{(this->*parse_and_set)(tokens.at(1), line_number)};
	if (!result) return result;

	*seen = true;
	return MakeConfigOk();
}

Config::ParseResult Config::HandleEpochs(const Tokens& tokens,
										 int64_t line_number) {
	return ParseSingleValue(&seen_epochs_, tokens, line_number,
							&Config::ParseEpochs);
}

Config::ParseResult Config::HandleLearningRate(const Tokens& tokens,
											   int64_t line_number) {
	return ParseSingleValue(&seen_learning_rate_, tokens, line_number,
							&Config::ParseLearningRate);
}

Config::ParseResult Config::HandleBatchSize(const Tokens& tokens,
											int64_t line_number) {
	return ParseSingleValue(&seen_batch_size_, tokens, line_number,
							&Config::ParseBatchSize);
}

Config::ParseResult Config::HandleLossFunc(const Tokens& tokens,
										   int64_t line_number) {
	return ParseSingleValue(&seen_loss_func_, tokens, line_number,
							&Config::ParseLossFunc);
}

Config::ParseResult Config::ParseEpochs(std::string_view token,
										int64_t line_number) {
	int64_t value{};
	if (!config_utils::ParseSigned(token, &value) || value <= 0) {
		return MakeConfigError(ConfigError::Code::kInvalidEpochsValue,
							   line_number);
	}
	epochs_ = value;
	return MakeConfigOk();
}

Config::ParseResult Config::ParseLearningRate(std::string_view token,
											  int64_t line_number) {
	float value{};
	if (!config_utils::ParseFloat(token, &value) || value <= 0.0F) {
		return MakeConfigError(ConfigError::Code::kInvalidLearningRateValue,
							   line_number);
	}
	learning_rate_ = value;
	return MakeConfigOk();
}

Config::ParseResult Config::ParseLossFunc(std::string_view token,
										  int64_t line_number) {
	auto result = ALoss::LossType(token);
	if (!result)
		return MakeConfigError(ConfigError::Code::kInvalidLossFuncValue,
							   line_number);
	loss_func_ = result.value();
	return MakeConfigOk();
}

Config::ParseResult Config::ParseBatchSize(std::string_view token,
										   int64_t line_number) {
	int64_t value{};
	if (!config_utils::ParseSigned(token, &value) || value <= 0) {
		return MakeConfigError(ConfigError::Code::kInvalidBatchSizeValue,
							   line_number);
	}
	batch_size_ = value;
	return MakeConfigOk();
}

Config::ParseResult Config::ParseLayer(const Tokens& tokens,
									   int64_t line_number) {
	if (tokens.size() != 3) {
		return MakeConfigError(ConfigError::Code::kInvalidLayerFormat,
							   line_number);
	}
	int64_t output_size{};
	if (!config_utils::ParseSigned(tokens.at(1), &output_size) ||
		output_size <= 0) {
		return MakeConfigError(ConfigError::Code::kInvalidLayerSize,
							   line_number);
	}
	auto result = AActivation::ActivationType(tokens.at(2));
	if (!result)
		return MakeConfigError(ConfigError::Code::kInvalidActivationFunc,
							   line_number);
	neuronal_layers_.push_back(output_size);
	activation_func_.push_back(result.value());
	size_++;
	return MakeConfigOk();
}

Config::Handlers Config::BuildHandlers() {
	return {
		{"epochs", &Config::HandleEpochs},
		{"learning_rate", &Config::HandleLearningRate},
		{"batch_size", &Config::HandleBatchSize},
		{"loss", &Config::HandleLossFunc},
		{"layer", &Config::ParseLayer},
	};
}

Config::ParseResult Config::ParseLines(std::ifstream& file) {
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
			return MakeConfigError(ConfigError::Code::kUnknownKey, line_number);
		}
		auto result{(this->*it->second)(tokens, line_number)};
		if (!result) return result;
	}

	auto validity{IsConfigValid()};
	if (!validity) return validity;

	return MakeConfigOk();
}

Config::ParseResult Config::IsConfigValid() const {
	if (!seen_epochs_ || epochs_ <= 0)
		return MakeConfigError(ConfigError::Code::kMissingEpochs);
	if (!seen_batch_size_ || batch_size_ <= 0)
		return MakeConfigError(ConfigError::Code::kMissingBatchSize);
	if (!seen_learning_rate_ || !std::isfinite(learning_rate_) ||
		learning_rate_ <= 0.0F)
		return MakeConfigError(ConfigError::Code::kMissingLearningRate);
	if (!seen_loss_func_)
		return MakeConfigError(ConfigError::Code::kMissingLossFunc);
	if (neuronal_layers_.empty() || size_ <= 0)
		return MakeConfigError(ConfigError::Code::kMissingLayer);
	if (activation_func_.size() != static_cast<size_t>(size_))
		return MakeConfigError(
			ConfigError::Code::kActivationLayerCountMismatch);
	if (neuronal_layers_.size() != static_cast<size_t>(size_))
		return MakeConfigError(ConfigError::Code::kLayerCountMismatch);
	if (std::any_of(neuronal_layers_.cbegin(), neuronal_layers_.cend(),
					[](int64_t layer_size) { return layer_size == 0; }))
		return MakeConfigError(ConfigError::Code::kZeroLayerSize);
	return MakeConfigOk();
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

std::string ConfigErrorMessage(const ConfigError& error) {
	std::string message{"Config parse error: "};
	if (error.line_number > 0) {
		message += "line ";
		message += std::to_string(error.line_number);
		message += ": ";
	}

	switch (error.code) {
		case ConfigError::Code::kCannotOpen:
			message += "cannot open file";
			break;
		case ConfigError::Code::kDuplicateEntry:
			message += "duplicate entry";
			break;
		case ConfigError::Code::kInvalidFormat:
			message += "invalid format";
			break;
		case ConfigError::Code::kInvalidEpochsValue:
			message += "invalid epochs value";
			break;
		case ConfigError::Code::kInvalidLearningRateValue:
			message += "invalid learning rate value";
			break;
		case ConfigError::Code::kInvalidBatchSizeValue:
			message += "invalid batch size value";
			break;
		case ConfigError::Code::kInvalidInputSizeValue:
			message += "invalid input size value";
			break;
		case ConfigError::Code::kInvalidLossFuncValue:
			message += "invalid loss function value";
			break;
		case ConfigError::Code::kInvalidLayerFormat:
			message += "invalid layer format";
			break;
		case ConfigError::Code::kInvalidLayerSize:
			message += "invalid layer size";
			break;
		case ConfigError::Code::kInvalidActivationFunc:
			message += "invalid activation function";
			break;
		case ConfigError::Code::kUnknownKey:
			message += "unknown key";
			break;
		case ConfigError::Code::kMissingEpochs:
			message += "missing epochs";
			break;
		case ConfigError::Code::kMissingBatchSize:
			message += "missing batch size";
			break;
		case ConfigError::Code::kMissingLearningRate:
			message += "missing learning rate";
			break;
		case ConfigError::Code::kMissingLossFunc:
			message += "missing loss function";
			break;
		case ConfigError::Code::kMissingLayer:
			message += "missing layer definition";
			break;
		case ConfigError::Code::kLayerCountMismatch:
			message += "layer count mismatch";
			break;
		case ConfigError::Code::kActivationLayerCountMismatch:
			message += "activation layer count mismatch";
			break;
		case ConfigError::Code::kZeroLayerSize:
			message += "layer size cannot be zero";
			break;
		case ConfigError::Code::kIncompleteConfiguration:
			message += "incomplete or invalid configuration";
			break;
	}

	return message;
}
