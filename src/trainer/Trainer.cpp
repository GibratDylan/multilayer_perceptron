#include "trainer/Trainer.hpp"

#include "trainer/observer/ITrainerObserver.hpp"

#include <vector>

void Trainer::add_observer(const ITrainerObserver* observer) {
	assert(observer != nullptr);

	observers_.push_back(observer);
}

