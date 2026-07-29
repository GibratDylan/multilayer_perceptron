#pragma once

#include <utility>
#include <variant>

template <class T, class E>
class Result {
   public:
	explicit Result(T value) : data_{std::move(value)} {}

	explicit Result(E error) : data_{std::move(error)} {}

	explicit operator bool() const noexcept {
		return std::holds_alternative<T>(data_);
	}

	// NOLINTBEGIN(readability-identifier-naming)
	const T& value() const { return std::get<T>(data_); }

	const E& error() const { return std::get<E>(data_); }

	// NOLINTEND(readability-identifier-naming)

   private:
	std::variant<T, E> data_;
};
