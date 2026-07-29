#include "trainer/observer/TrainerObserverMetricsLog.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>

void TrainerObserverMetricsLog::OnEpochEnd(int64_t epoch_index, float loss,
										   float accuracy) {
	static bool header_printed{};

	constexpr int k_border_first_segment{10};
	constexpr int k_border_second_segment{16};
	constexpr int k_border_third_segment{14};
	constexpr int k_epoch_width{8};
	constexpr int k_loss_width{14};
	constexpr int k_accuracy_width{12};
	constexpr int k_precision{6};

	constexpr const char* reset = "\033[0m";
	constexpr const char* bold = "\033[1m";
	constexpr const char* cyan = "\033[36m";
	constexpr const char* green = "\033[32m";
	constexpr const char* yellow = "\033[33m";
	constexpr const char* magenta = "\033[35m";
	constexpr const char* dim = "\033[2m";

	auto print_border{[&] {
		std::cout << dim << '+' << std::string(k_border_first_segment, '-')
				  << '+' << std::string(k_border_second_segment, '-') << '+'
				  << std::string(k_border_third_segment, '-') << '+' << reset
				  << '\n';
	}};

	if (!header_printed) {
		std::cout << bold << cyan << "Training metrics" << reset << '\n';
		print_border();
		std::cout << dim << '|' << reset << ' ' << bold << std::left
				  << std::setw(k_epoch_width) << "Epoch" << reset << ' ' << dim
				  << '|' << reset << ' ' << bold << std::left
				  << std::setw(k_loss_width) << "Loss" << reset << ' ' << dim
				  << '|' << reset << ' ' << bold << std::left
				  << std::setw(k_accuracy_width) << "Accuracy" << reset << ' '
				  << dim << '|' << reset << '\n';
		print_border();
		header_printed = true;
	}

	std::cout << dim << '|' << reset << ' ' << yellow << std::left
			  << std::setw(k_epoch_width) << epoch_index << reset << ' ' << dim
			  << '|' << reset << ' ' << green << std::right << std::fixed
			  << std::setprecision(k_precision) << std::setw(k_loss_width)
			  << loss << reset << ' ' << dim << '|' << reset << ' ' << magenta
			  << std::right << std::fixed << std::setprecision(k_precision)
			  << std::setw(k_accuracy_width) << accuracy << reset << ' ' << dim
			  << '|' << reset << '\n';
	print_border();
}
