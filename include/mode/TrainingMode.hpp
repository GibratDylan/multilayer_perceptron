#pragma once

#include "mode/IProgramMode.hpp"

#include <string_view>

class TrainingMode : public IProgramMode {
   private:
	std::string_view training_data_path_{};

   public:
	explicit TrainingMode(std::string_view training_data_path);

	void Run() const override;
};
