#pragma once

#include <cstdint>

struct ConfigError {
	enum class Code : uint8_t {
		kCannotOpen,
		kDuplicateEntry,
		kInvalidFormat,
		kInvalidEpochsValue,
		kInvalidLearningRateValue,
		kInvalidBatchSizeValue,
		kInvalidInputSizeValue,
		kInvalidLossFuncValue,
		kInvalidLayerFormat,
		kInvalidLayerSize,
		kInvalidActivationFunc,
		kUnknownKey,
		kMissingEpochs,
		kMissingBatchSize,
		kMissingLearningRate,
		kMissingLossFunc,
		kMissingLayer,
		kLayerCountMismatch,
		kActivationLayerCountMismatch,
		kZeroLayerSize,
		kIncompleteConfiguration,
	};

	Code code{};
	int64_t line_number{};
};

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
