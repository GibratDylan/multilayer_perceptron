#include "mode/PredictiveMode.hpp"

#include <string_view>

PredictiveMode::PredictiveMode(std::string_view predictive_data_path,
							   std::string_view serialized_model_path)
	: predictive_data_path_{predictive_data_path},
	  serialized_model_path_{serialized_model_path} {}

void PredictiveMode::Run() const {}
