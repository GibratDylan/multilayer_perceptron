#pragma once

#include "observer/ITrainerObserver.hpp"

#include <vector>

class Trainer {
   private:
	std::vector<const ITrainerObserver*> _observer;

   public:
	void add_observer(const ITrainerObserver* observer);
};
