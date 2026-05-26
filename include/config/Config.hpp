#pragma once

#include <string>
#include <string_view>
#include <vector>

class Config {
   private:
	std::vector<int> neuronal_layers_;
	std::vector<std::string> activation_func_;
	int epochs_;
	int batch_size_;
	float learning_rate_;
	std::string loss_func_;

	std::string path_;

   public:
	Config(std::string_view path);

	bool parse();

	bool getNeuralLayer() const;
	bool getActivationFunc() const;
	bool getEpochs() const;
	bool getLearningRate() const;
	bool getLossFunc() const;
   bool getBatchSize() const;

   private:
	static bool isNeuralLayerValid(std::string_view string);
	static bool isActivationFuncValid(std::string_view string);
	static bool isEpochsValid(std::string_view string);
	static bool isLearningRateValid(std::string_view string);
	static bool isLossFuncValid(std::string_view string);
	static bool isBatchSizeValid(std::string_view string);
};
