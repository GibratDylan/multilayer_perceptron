#pragma once

#include <ostream>

#include <Eigen/Dense>

namespace activationFuncString {
const std::string RELU = "relu";
const std::string SOFTMAX = "softmax";
};	// namespace activationFuncString

class AActivation {
   public:
	enum activationFuncType : uint8_t { RELU, SOFTMAX, NONE };

   protected:
	Eigen::MatrixXd outputs_;
	Eigen::MatrixXd inputs_;

	Eigen::MatrixXd inputs_gradient_;

   public:
	virtual ~AActivation() = default;

	virtual void forward(const Eigen::MatrixXd& inputs) = 0;
	virtual void backward(const Eigen::MatrixXd& inputs) = 0;

	const Eigen::MatrixXd& getOutputs() const;
	const Eigen::MatrixXd& getInputsGradient() const;

	static AActivation::activationFuncType getActivationType(
		std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const AActivation& rhs);
};
