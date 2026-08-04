#pragma once

#include "mode/IProgramMode.hpp"

#include <string_view>

class SplitMode : public IProgramMode {
   private:
	std::string_view split_data_path_{};
	float split_ratio_{};

   public:
	explicit SplitMode(std::string_view split_data_path, float split_ratio);

	void Run() const override;
};
