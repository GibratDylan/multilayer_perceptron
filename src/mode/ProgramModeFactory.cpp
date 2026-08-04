#include "mode/ProgramModeFactory.hpp"

#include "error/Result.hpp"
#include "error/error.hpp"
#include "mode/IProgramMode.hpp"
#include "mode/PredictiveMode.hpp"
#include "mode/SplitMode.hpp"
#include "mode/TrainingMode.hpp"

#include <charconv>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

ProgramModeFactory::ProgramModeFactory(
	const std::vector<std::string_view>& args) {
	if (args.size() < 3 || args.size() > 4)
		throw std::runtime_error(std::string{kCommandsString});

	auto result{GetMode(args.at(1))};
	if (result) {
		mode_ = result.value();
	} else {
		throw std::runtime_error(std::string{kCommandsString});
	}
}

Result<std::unique_ptr<IProgramMode>, ProgramModeError>
ProgramModeFactory::Create(const std::vector<std::string_view>& args) const {
	switch (mode_) {
		case ProgramModeType::kPredicitive: {
			if (args.size() != 4)
				return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
					ProgramModeError::kNumberOfArgumentsNotValid};
			return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
				std::make_unique<PredictiveMode>(args.at(2), args.at(3))};
		}
		case ProgramModeType::kTraining: {
			if (args.size() != 4)
				return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
					ProgramModeError::kNumberOfArgumentsNotValid};
			return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
				std::make_unique<TrainingMode>(args.at(2), args.at(3))};
		}
		case ProgramModeType::kSplit: {
			if (args.size() != 4)
				return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
					ProgramModeError::kNumberOfArgumentsNotValid};

			float split_ratio{};
			auto result{std::from_chars(args.at(3).begin(), args.at(3).end(),
										split_ratio)};
			if (result.ec != std::errc{}) {
				return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
					ProgramModeError::kValueNotValid};
			}

			return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
				std::make_unique<SplitMode>(args.at(2), split_ratio)};
		}
		default:
			return Result<std::unique_ptr<IProgramMode>, ProgramModeError>{
				ProgramModeError::kModeNotValid};
	}
}

Result<ProgramModeFactory::ProgramModeType, ProgramModeError>
ProgramModeFactory::GetMode(std::string_view str) noexcept {
	if (str == program_mode_string::kPredicitive)
		return Result<ProgramModeType, ProgramModeError>{
			ProgramModeType::kPredicitive};
	if (str == program_mode_string::kTraining)
		return Result<ProgramModeType, ProgramModeError>{
			ProgramModeType::kTraining};
	if (str == program_mode_string::kSplit)
		return Result<ProgramModeType, ProgramModeError>{
			ProgramModeType::kSplit};
	return Result<ProgramModeType, ProgramModeError>{
		ProgramModeError::kModeNotValid};
}
