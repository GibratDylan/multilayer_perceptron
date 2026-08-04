#include "mode/ProgramModeFactory.hpp"

#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

int main(const int argc, const char** argv) {
	try {
		// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		std::vector<std::string_view> args(argv, argv + argc);
		// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

		ProgramModeFactory factory{args};

		auto result = factory.Create(args);
		if (result) {
			result.value()->Run();
		} else {
			return 1;
		}

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
