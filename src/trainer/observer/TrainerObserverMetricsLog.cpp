#include "trainer/observer/TrainerObserverMetricsLog.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

void TrainerObserverMetricsLog::OnEpochEnd(int64_t epoch_index, float loss,
										   float accuracy) {
	static bool header_printed{false};

	constexpr const char* kReset = "\033[0m";
	constexpr const char* kBold = "\033[1m";
	constexpr const char* kCyan = "\033[36m";
	constexpr const char* kGreen = "\033[32m";
	constexpr const char* kYellow = "\033[33m";
	constexpr const char* kMagenta = "\033[35m";
	constexpr const char* kDim = "\033[2m";

	auto print_border{[] {
		std::cout << kDim << '+' << std::string(10, '-') << '+'
				  << std::string(16, '-') << '+' << std::string(14, '-') << '+'
				  << kReset << '\n';
	}};

	if (!header_printed) {
		std::cout << kBold << kCyan << "Training metrics" << kReset << '\n';
		print_border();
		std::cout << kDim << '|' << kReset << ' ' << kBold << std::left
				  << std::setw(8) << "Epoch" << kReset << ' ' << kDim << '|'
				  << kReset << ' ' << kBold << std::left << std::setw(14)
				  << "Loss" << kReset << ' ' << kDim << '|' << kReset << ' '
				  << kBold << std::left << std::setw(12) << "Accuracy" << kReset
				  << ' ' << kDim << '|' << kReset << '\n';
		print_border();
		header_printed = true;
	}

	std::cout << kDim << '|' << kReset << ' ' << kYellow << std::left
			  << std::setw(8) << epoch_index << kReset << ' ' << kDim << '|'
			  << kReset << ' ' << kGreen << std::right << std::fixed
			  << std::setprecision(6) << std::setw(14) << loss << kReset << ' '
			  << kDim << '|' << kReset << ' ' << kMagenta << std::right
			  << std::fixed << std::setprecision(6) << std::setw(12) << accuracy
			  << kReset << ' ' << kDim << '|' << kReset << '\n';
	print_border();
}
