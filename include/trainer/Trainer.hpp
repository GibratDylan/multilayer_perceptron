#pragma once

#include "observer/ITrainerObserver.hpp"

#include <vector>

class Trainer {
   private:
	std::vector<ITrainerObserver*> _observer;

   public:
	void add_observer();
};
