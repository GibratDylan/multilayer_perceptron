#pragma once

#include "activation/AActivation.hpp"
#include "loss/ALoss.hpp"

#include <string>
#include <string_view>
#include <vector>

class Config {
   private:
	unsigned int size_;
	std::vector<unsigned int> neuronal_layers_;
	std::vector<AActivation::activationFuncType> activation_func_;
	unsigned int epochs_;
	unsigned int batch_size_;
	float learning_rate_;
	ALoss::lossFuncType loss_func_;

	std::string path_;

   public:
	Config(std::string_view path);

	bool parse();
	bool isConfigValid() const;

	const std::vector<unsigned int>& getNeuralLayer() const;
	const std::vector<AActivation::activationFuncType>& getActivationFunc()
		const;
	unsigned int getSize() const;
	unsigned int getEpochs() const;
	unsigned int getBatchSize() const;
	float getLearningRate() const;
	ALoss::lossFuncType getLossFunc() const;

   private:
	static bool isActivationFuncValid(std::string_view string);
	static bool isLossFuncValid(std::string_view string);
};
