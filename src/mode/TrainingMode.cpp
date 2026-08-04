#include "mode/TrainingMode.hpp"

#include "config/Config.hpp"
#include "data/Dataset.hpp"
#include "data/csv.hpp"
#include "network/Network.hpp"
#include "network/NetworkBuilder.hpp"
#include "trainer/Trainer.hpp"
#include "trainer/observer/TrainerObserverMetricsCsv.hpp"
#include "trainer/observer/TrainerObserverMetricsLog.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

TrainingMode::TrainingMode(std::string_view training_data_path,
						   std::string_view config_path)
	: training_data_path_{training_data_path}, config_path_{config_path} {}

void TrainingMode::Run() const {
	Config config{std::string{config_path_}};
	auto config_result{config.Parse()};
	if (!config_result) {
		throw std::runtime_error(ConfigErrorMessage(config_result.error()));
	}

	auto dataset_result{csv::CsvLoader(std::string{training_data_path_})};
	if (!dataset_result) {
		throw std::runtime_error(
			std::string(csv::CsvErrorMessage(dataset_result.error())));
	}

	Dataset dataset{std::move(dataset_result.value())};

	NetworkBuilder network_builder{config, dataset.GetSizeRows()};

	Network network{network_builder.Build()};

	Trainer trainer{};

	trainer.AddObserver(std::make_unique<TrainerObserverMetricsCsv>());
	trainer.AddObserver(std::make_unique<TrainerObserverMetricsLog>());

	trainer.Train(network, dataset, config);

	network.Serialize("test");
}
