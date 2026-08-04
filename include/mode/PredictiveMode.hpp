#pragma once

#include "mode/IProgramMode.hpp"

#include <string_view>

class PredictiveMode : public IProgramMode {
   private:
	std::string_view predictive_data_path_{};
	std::string_view serialized_model_path_{};

   public:
	explicit PredictiveMode(std::string_view predictive_data_path,
							std::string_view serialized_model_path);

	void Run() const override;
};
