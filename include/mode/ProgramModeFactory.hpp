#pragma once

#include "IProgramMode.hpp"
#include "error/Result.hpp"
#include "error/error.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace program_mode_string {
inline constexpr std::string_view kPredicitive{"--predictive"};
inline constexpr std::string_view kTraining{"--training"};
inline constexpr std::string_view kSplit{"--split"};
}  // namespace program_mode_string

class ProgramModeFactory {
   public:
	enum class ProgramModeType : uint8_t {
		kPredicitive,
		kTraining,
		kSplit,
	};

   public:
	explicit ProgramModeFactory(const std::vector<std::string_view>& args);

	Result<std::unique_ptr<IProgramMode>, ProgramModeError> Create(
		const std::vector<std::string_view>& args) const;

   private:
	static Result<ProgramModeType, ProgramModeError> GetMode(
		std::string_view str) noexcept;

   private:
	ProgramModeType mode_{};

	static constexpr std::string_view kCommandsString{
		"Commands:\n--predictive <model_serialized> "
		"<predictive_data> <config_file>\n--training "
		"<training_data> <config_file>\n--split <data> <ratio>\n--show "
		"<metrics>"};
};
