#include "trainer/observer/TrainerObserverMetricsWriter.hpp"

#include <fstream>
#include <stdexcept>

#include <Eigen/Dense>

void TrainerObserverMetricsWriter::on_epoch_end(int epoch_index,
												float accuracy) {
	std::ofstream metrics_csv{"metrics.csv", std::ios::out | std::ios::app};
	if (!metrics_csv.good()) {
		throw std::runtime_error("std::ofstream fail");
	}
	if (!metrics_csv.tellp()) {
		metrics_csv << "epoch_index" << ',' << "accuracy" << '\n';
	}
	metrics_csv << epoch_index << ',' << accuracy << '\n';
}
