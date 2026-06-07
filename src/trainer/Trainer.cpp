#include "trainer/Trainer.hpp"

#include "config/Config.hpp"
#include "data/Dataset.hpp"
#include "network/Network.hpp"
#include "trainer/metrics.hpp"
#include "trainer/observer/ITrainerObserver.hpp"

#include <cassert>
#include <vector>

void Trainer::AddObserver(std::unique_ptr<ITrainerObserver>&& observer) {
	assert(observer != nullptr);

	observers_.emplace_back(std::move(observer));
}

void Trainer::Train(Network& network, Dataset& dataset,
					const Config& config) const {
	for (int64_t epoch{1}; epoch <= config.GetEpochs(); ++epoch) {
		dataset.RandDataset();

		float accuracy{};
		float loss{};
		for (auto&& [batch, target] : dataset) {
			network.ForwardPass(batch, target);
			network.BackwardPass(target);

			loss = network.GetLossFunc().GetLoss();
			accuracy = metrics::Accuracy(batch, target);
		}

		for (const auto& observer : observers_) {
			observer->OnEpochEnd(epoch, loss, accuracy);
		}
	}
}
