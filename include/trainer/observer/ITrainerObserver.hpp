#pragma once

#include <cstdint>

class ITrainerObserver {
   public:
	explicit ITrainerObserver(const ITrainerObserver&) = delete;
	explicit ITrainerObserver(ITrainerObserver&&) noexcept = delete;
	ITrainerObserver& operator=(const ITrainerObserver&) = delete;
	ITrainerObserver& operator=(ITrainerObserver&&) noexcept = delete;

	virtual ~ITrainerObserver() = default;
	virtual void OnEpochEnd(int64_t epoch_index, float loss,
							float accuracy) = 0;

   protected:
	explicit ITrainerObserver() = default;
};
