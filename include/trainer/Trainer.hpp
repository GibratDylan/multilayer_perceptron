#pragma once

#include "observer/ITrainerObserver.hpp"

#include <vector>

class Trainer {
   private:
	std::vector<const ITrainerObserver*> observers_;

   public:
	void AddObserver(const ITrainerObserver* observer);
};
