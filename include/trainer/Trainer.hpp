#pragma once

#include "config/Config.hpp"
#include "data/Dataset.hpp"
#include "network/Network.hpp"
#include "observer/ITrainerObserver.hpp"

#include <memory>
#include <vector>

class Trainer {
   private:
	std::vector<std::unique_ptr<ITrainerObserver>> observers_{};

   public:
	void AddObserver(std::unique_ptr<ITrainerObserver>&& observer);
	void Train(Network& network, Dataset& dataset, const Config& config) const;
};
