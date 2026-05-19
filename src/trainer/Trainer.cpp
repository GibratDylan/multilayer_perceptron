#include "trainer/Trainer.hpp"

#include "trainer/observer/ITrainerObserver.hpp"

#include <vector>

void Trainer::add_observer(const ITrainerObserver* observer) {
    assert(observer != nullptr);
    _observer.push_back(observer);
}
