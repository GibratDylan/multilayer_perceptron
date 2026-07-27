#include "trainer/observer/TrainerObserverMetricsLog.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

void TrainerObserverMetricsLog::OnEpochEnd(int64_t epoch_index, float loss,
										   float accuracy) {
	static bool header_printed{false};

	constexpr int kBorderFirstSegment{10};
	constexpr int kBorderSecondSegment{16};
	constexpr int kBorderThirdSegment{14};
	constexpr int kEpochWidth{8};
	constexpr int kLossWidth{14};
	constexpr int kAccuracyWidth{12};
	constexpr int kPrecision{6};

	constexpr const char* reset = "\033[0m";
	constexpr const char* bold = "\033[1m";
	constexpr const char* cyan = "\033[36m";
	constexpr const char* green = "\033[32m";
	constexpr const char* yellow = "\033[33m";
	constexpr const char* magenta = "\033[35m";
	constexpr const char* dim = "\033[2m";

	auto print_border{[&] {
		std::cout << dim << '+' << std::string(kBorderFirstSegment, '-') << '+'
				  << std::string(kBorderSecondSegment, '-') << '+'
				  << std::string(kBorderThirdSegment, '-') << '+' << reset
				  << '\n';
	}};

	if (!header_printed) {
		std::cout << bold << cyan << "Training metrics" << reset << '\n';
		print_border();
		std::cout << dim << '|' << reset << ' ' << bold << std::left
				  << std::setw(kEpochWidth) << "Epoch" << reset << ' ' << dim
				  << '|' << reset << ' ' << bold << std::left
				  << std::setw(kLossWidth) << "Loss" << reset << ' ' << dim
				  << '|' << reset << ' ' << bold << std::left
				  << std::setw(kAccuracyWidth) << "Accuracy" << reset << ' '
				  << dim << '|' << reset << '\n';
		print_border();
		header_printed = true;
	}

	std::cout << dim << '|' << reset << ' ' << yellow << std::left
			  << std::setw(kEpochWidth) << epoch_index << reset << ' ' << dim
			  << '|' << reset << ' ' << green << std::right << std::fixed
			  << std::setprecision(kPrecision) << std::setw(kLossWidth) << loss
			  << reset << ' ' << dim << '|' << reset << ' ' << magenta
			  << std::right << std::fixed << std::setprecision(kPrecision)
			  << std::setw(kAccuracyWidth) << accuracy << reset << ' ' << dim
			  << '|' << reset << '\n';
	print_border();
}
