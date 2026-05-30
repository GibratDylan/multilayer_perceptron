#include "trainer/Trainer.hpp"

#include "trainer/observer/ITrainerObserver.hpp"

#include <vector>
#include <cassert>

void Trainer::AddObserver(const ITrainerObserver* observer) {
	assert(observer != nullptr);

	observers_.push_back(observer);
}
