#include "mode/TrainingMode.hpp"

#include <string_view>

TrainingMode::TrainingMode(std::string_view training_data_path)
	: training_data_path_{training_data_path} {}

void TrainingMode::Run() const {}
