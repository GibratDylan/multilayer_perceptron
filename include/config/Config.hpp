#pragma once

#include "activation/AActivation.hpp"
#include "loss/ALoss.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class Config {
   public:
	using Tokens = std::vector<std::string>;
	using Handler = bool (Config::*)(const Tokens&, int64_t);
	using Handlers = std::unordered_map<std::string_view, Handler>;
	using ParseSingle = bool (Config::*)(std::string_view, int64_t);

   private:
	int64_t size_;
	std::vector<int64_t> neuronal_layers_;
	std::vector<AActivation::ActivationFuncType> activation_func_;
	int64_t epochs_;
	int64_t batch_size_;
	float learning_rate_;
	ALoss::LossFuncType loss_func_;

	std::string path_;

	bool seen_epochs_;
	bool seen_learning_rate_;
	bool seen_batch_size_;
	bool seen_loss_;

   public:
	explicit Config(std::string_view path);

	bool Parse();

	const std::vector<int64_t>& GetNeuralLayer() const;
	const std::vector<AActivation::ActivationFuncType>& GetActivationFunc()
		const;
	int64_t GetSize() const;
	int64_t GetEpochs() const;
	int64_t GetBatchSize() const;
	float GetLearningRate() const;
	ALoss::LossFuncType GetLossFunc() const;

   private:
	void Reset();
	bool ParseLines(std::ifstream& file);
	bool IsConfigValid() const;

	bool ParseSingleValue(bool* seen, const Tokens& tokens, int64_t line_number,
						  ParseSingle parse_and_set);
	bool HandleEpochs(const Tokens& tokens, int64_t line_number);
	bool HandleLearningRate(const Tokens& tokens, int64_t line_number);
	bool HandleBatchSize(const Tokens& tokens, int64_t line_number);
	bool HandleLoss(const Tokens& tokens, int64_t line_number);
	bool ParseEpochs(std::string_view token, int64_t line_number);
	bool ParseLearningRate(std::string_view token, int64_t line_number);
	bool ParseLoss(std::string_view token, int64_t line_number);
	bool ParseBatchSize(std::string_view token, int64_t line_number);
	bool ParseLayer(const Tokens& tokens, int64_t line_number);

	Handlers BuildHandlers();

	static bool IsActivationFuncValid(std::string_view string);
	static bool IsLossFuncValid(std::string_view string);
};
