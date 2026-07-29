#include "data/Dataset.hpp"
#include "data/csv.hpp"

#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

// class ProgramType
// --predictive <model_serialized> <predictive_data>
// --training <training_data>
// --split <data>
// --show <metrics>
int main(const int argc, const char** argv) {
	try {
		// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		std::vector<std::string_view> args(argv, argv + argc);
		// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

		if (args.size() != 2) {
			std::cerr << "Program need config file: ./mlp path/to/config.txt\n";
			return 1;
		}

		// Config config{std::string{argv[1]}};
		// if (!config.Parse()) return 1;

		// /// inputs = rows(input), batch = cols(batch)
		// Matrix inputs(config.GetNeuralLayer().front(),
		// config.GetBatchSize()); for (int64_t i{}; i < inputs.rows(); ++i) {
		// 	for (int64_t j{}; j < inputs.cols(); ++j) {
		// 		inputs(i, j) = 0.5F;
		// 	}
		// }

		// /// targets = rows(batch)
		// IntVector targets(config.GetBatchSize());
		// for (int64_t i{}; i < targets.rows(); ++i) targets(i) = 1;

		// Dataset dataset{std::move(inputs), std::move(targets)};

		// NetworkBuilder network_builder{config};

		// Network network{network_builder.Build()};

		// Trainer trainer{};

		// trainer.AddObserver(std::make_unique<TrainerObserverMetricsCsv>());
		// trainer.AddObserver(std::make_unique<TrainerObserverMetricsLog>());

		// trainer.Train(network, dataset, config);

		Dataset test{csv::CsvLoader(std::string{args.at(1)})};

		constexpr float k_split_ratio{0.1F};
		auto pair{csv::DatasetSplit(test, k_split_ratio)};

		csv::CsvDumper("data/test1.csv", "", pair.first);
		csv::CsvDumper("data/test2.csv", "", pair.second);

	} catch (std::exception& e) {
		std::cerr << e.what() << '\n';
		return 1;
	}

	return 0;
}

// Chercher erreur optimisation
// Optimise memory usage (Ne plus stocker dans les backwardpass ? )
// Optimizer
// Ajouter des metrics et modifier les string en dur dans MetricsCsv (Les
// recuperer depuis main)
// Rajouter et verifier les assert Catch exception main
// et exception safety program Release mode
