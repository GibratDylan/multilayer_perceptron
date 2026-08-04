#pragma once

#include "config/Config.hpp"
#include "network/Network.hpp"

#include <cstdint>

class NetworkBuilder {
   private:
	Config config_;
	int64_t input_size_data_{};

   public:
	explicit NetworkBuilder(Config config, int64_t input_size_data);

	Network Build() const;
};
