#pragma once

class IProgramMode {
   public:
	explicit IProgramMode(const IProgramMode&) = delete;
	explicit IProgramMode(IProgramMode&&) noexcept = delete;
	IProgramMode& operator=(const IProgramMode&) = delete;
	IProgramMode& operator=(IProgramMode&&) noexcept = delete;
	virtual ~IProgramMode() = default;

	virtual void Run() const = 0;

   protected:
	IProgramMode() = default;
};
