#include "data/Dataset.hpp"
#include "data/csv.hpp"

#include <iostream>
#include <string>

int main(int argc, const char** argv) {
	if (argc != 2) {
		std::cerr << "Program need config file: ./mlp path/to/config.txt\n";
		return 1;
	}

	// Config config{std::string{argv[1]}};
	// if (!config.Parse()) return 1;

	// /// inputs = rows(input), batch = cols(batch)
	// Matrix inputs(config.GetNeuralLayer().front(), config.GetBatchSize());
	// for (int64_t i{}; i < inputs.rows(); ++i) {
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

	Dataset test{csv::CsvLoader(argv[1])};

	auto pair{csv::DatasetSplit(test, 0.1)};

	csv::CsvDumper("test1", "", pair.first);
	csv::CsvDumper("test2", "", pair.second);

	return 0;
}

// Chercher erreur optimisation
// Optimise memory usage (Ne plus stocker dans les backwardpass ? )
// BackwardPass
// Optimizer
// Ajouter des metrics et modifier les string en dur dans MetricsCsv (Les
// recuperer depuis main)
// Rajouter et verifier les assert Catch exception main
// et exception safety program Release mode
