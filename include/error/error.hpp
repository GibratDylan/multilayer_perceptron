#pragma once

#include <cstdint>

enum class CsvError : uint8_t {
	kCannotOpen,
	kCannotCreate,
	kCannotWrite,
	kEmpty,
	kNotRectangular,
	kBadFloat,
};

enum class ProgramModeError : uint8_t {
	kModeNotValid,
	kNumberOfArgumentsNotValid,
	kValueNotValid,
};

enum class ActivationFuncError : uint8_t {
	kNotValidActivationFunc,
};

enum class LossFuncError : uint8_t {
	kNotValidLoss,
};
