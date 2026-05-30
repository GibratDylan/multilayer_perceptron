#pragma once

#include "activation/AActivation.hpp"
#include "loss/ALoss.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class Config;

using Tokens = std::vector<std::string>;
using Handler = bool (Config::*)(const Tokens&, size_t);
using Handlers = std::unordered_map<std::string_view, Handler>;
using ParseSingle = bool (Config::*)(std::string_view, size_t);

class Config {
   private:
	size_t size_;
	std::vector<size_t> neuronal_layers_;
	std::vector<AActivation::activationFuncType> activation_func_;
	size_t epochs_;
	size_t batch_size_;
	float learning_rate_;
	ALoss::lossFuncType loss_func_;

	std::string path_;

	bool seen_epochs_;
	bool seen_learning_rate_;
	bool seen_batch_size_;
	bool seen_loss_;

   public:
	Config(std::string_view path);

	bool parse();

	const std::vector<size_t>& getNeuralLayer() const;
	const std::vector<AActivation::activationFuncType>& getActivationFunc()
		const;
	size_t getSize() const;
	size_t getEpochs() const;
	size_t getBatchSize() const;
	float getLearningRate() const;
	ALoss::lossFuncType getLossFunc() const;

   private:
	void reset();
	bool parseLines(std::ifstream& file);
	bool isConfigValid() const;

	bool parseSingleValue(bool* seen, const Tokens& tokens, size_t line_number,
						  ParseSingle parse_and_set);
	bool handleEpochs(const Tokens& tokens, size_t line_number);
	bool handleLearningRate(const Tokens& tokens, size_t line_number);
	bool handleBatchSize(const Tokens& tokens, size_t line_number);
	bool handleLoss(const Tokens& tokens, size_t line_number);
	bool parseEpochs(std::string_view token, size_t line_number);
	bool parseLearningRate(std::string_view token, size_t line_number);
	bool parseLoss(std::string_view token, size_t line_number);
	bool parseBatchSize(std::string_view token, size_t line_number);
	bool parseLayer(const Tokens& tokens, size_t line_number);

	Handlers build_handlers();

	static bool isActivationFuncValid(std::string_view string);
	static bool isLossFuncValid(std::string_view string);
};
