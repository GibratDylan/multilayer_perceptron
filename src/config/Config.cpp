#include "config/Config.hpp"

#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

Config::Config(std::string_view path) : path_(path) {}

bool Config::parse() {
	std::ifstream file(path_);
	if (!file) return false;

	std::string key;
	std::string value;
	const std::unordered_map<std::string, std::function<bool(std::string_view)>>
		handlers = {
			{"epochs",
			 [this](std::string_view v) {
				 if (!isEpochsValid(v)) return false;
				 epochs_ = std::stoi(std::string(v));
				 return true;
			 }},
			{"learning_rate",
			 [this](std::string_view v) {
				 if (!isLearningRateValid(v)) return false;
				 learning_rate_ = std::stof(std::string(v));
				 return true;
			 }},
			{"batch_size",
			 [this](std::string_view v) {
				 if (!isBatchSizeValid(v)) return false;
				 batch_size_ = std::stoi(std::string(v));
				 return true;
			 }},
		};

	while (file >> key) {
		if (!(file >> value)) return false;

		const auto it = handlers.find(key);
		if (it == handlers.end()) return false;
		if (!it->second(value)) return false;
	}

	return true;
}

// bool Config::getNeuralLayer() const {}

// bool Config::getActivationFunc() const {}

// bool Config::getEpochs() const {}

// bool Config::getLearningRate() const {}

// bool Config::getLossFunc() const {}

// bool Config::getBatchSize() const {}

// bool Config::isNeuralLayerValid(std::string_view string) {}

// bool Config::isActivationFuncValid(std::string_view string) {}

// bool Config::isEpochsValid(std::string_view string) {}

// bool Config::isLearningRateValid(std::string_view string) {}

// bool Config::isLossFuncValid(std::string_view string) {}

// bool Config::isBatchSizeValid(std::string_view string) {}
