#include "trainer/observer/TrainerObserverMetricsWriter.hpp"

#include <fstream>
#include <iostream>

// Rajouter d'autres metrics (Loss, learning rate, etc..) et les prints dans le
// terminal
void TrainerObserverMetricsWriter::OnEpochEnd(EpochIndex epoch_index,
											  AccuracyScore accuracy) {
	std::ofstream metrics_csv{"metrics.csv", std::ios::out | std::ios::app};

	if (!metrics_csv.good()) {
		std::cerr << "std::ofstream fail !\n";
		return;
	}

	if (!metrics_csv.tellp()) {
		metrics_csv << "epoch_index" << ',' << "accuracy" << '\n';
	}

	metrics_csv << epoch_index << ',' << accuracy << '\n';
}
