#pragma once

#include "config/Config.hpp"
#include "network/Network.hpp"

#include <memory>

class NetworkBuilder {
   private:
	const Config& config_;

   public:
	explicit NetworkBuilder(const Config& config);

	Network Build() const;

   private:
	std::unique_ptr<ALoss> GetLossFuncObj() const;
	std::unique_ptr<AActivation> GetActivationFuncObj(int64_t index) const;
};
