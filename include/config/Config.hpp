#pragma once

#include "activation/AActivation.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"
#include "loss/ALoss.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

class Config {
   public:
	using Tokens = std::vector<std::string>;
	using Handler =
		Result<std::monostate, ConfigError> (Config::*)(const Tokens&, int64_t);
	using Handlers = std::unordered_map<std::string_view, Handler>;
	using ParseSingle = Result<std::monostate, ConfigError> (Config::*)(
		std::string_view, int64_t);
	using ParseResult = Result<std::monostate, ConfigError>;

   private:
	int64_t size_{};
	std::vector<int64_t> neuronal_layers_{};
	std::vector<AActivation::ActivationFuncType> activation_func_{};
	int64_t epochs_{};
	int64_t batch_size_{};
	float learning_rate_{};
	ALoss::LossFuncType loss_func_{};

	std::string path_{};

	bool seen_epochs_{};
	bool seen_learning_rate_{};
	bool seen_batch_size_{};
	bool seen_loss_func_{};

   public:
	explicit Config(std::string_view path);

	ParseResult Parse();

	const std::vector<int64_t>& GetNeuralLayer() const noexcept;
	const std::vector<AActivation::ActivationFuncType>& GetActivationFunc()
		const noexcept;
	int64_t GetSize() const noexcept;
	int64_t GetEpochs() const noexcept;
	int64_t GetBatchSize() const noexcept;
	float GetLearningRate() const noexcept;
	ALoss::LossFuncType GetLossFunc() const noexcept;

	ParseResult IsConfigValid() const;

   private:
	void Reset();
	ParseResult ParseLines(std::ifstream& file);

	ParseResult ParseSingleValue(bool* seen, const Tokens& tokens,
								 int64_t line_number,
								 ParseSingle parse_and_set);

	ParseResult HandleEpochs(const Tokens& tokens, int64_t line_number);
	ParseResult HandleLearningRate(const Tokens& tokens, int64_t line_number);
	ParseResult HandleBatchSize(const Tokens& tokens, int64_t line_number);
	ParseResult HandleLossFunc(const Tokens& tokens, int64_t line_number);
	ParseResult ParseEpochs(std::string_view token, int64_t line_number);
	ParseResult ParseLearningRate(std::string_view token, int64_t line_number);
	ParseResult ParseLossFunc(std::string_view token, int64_t line_number);
	ParseResult ParseBatchSize(std::string_view token, int64_t line_number);
	ParseResult ParseLayer(const Tokens& tokens, int64_t line_number);

	Handlers BuildHandlers();
};

std::string ConfigErrorMessage(const ConfigError& error);
