#pragma once

#include "activation/AActivation.hpp"
#include "config/Config.hpp"
#include "loss/ALoss.hpp"
#include "network/Network.hpp"

#include <cstdint>
#include <memory>

class NetworkBuilder {
   private:
	const Config config_;

   public:
	explicit NetworkBuilder(Config config);

	Network Build() const;

   private:
	std::unique_ptr<ALoss> GetLossFuncObj() const;
	std::unique_ptr<AActivation> GetActivationFuncObj(int64_t index) const;
};
