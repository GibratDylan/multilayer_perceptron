#pragma once

#include <ostream>

#include <Eigen/Dense>

namespace lossFuncString {
const std::string CATCROSSENTROPY = "categorical_cross_entropy";
};

class ALoss {
   public:
	enum lossFuncType : uint8_t { CATCROSSENTROPY, NONE };

   protected:
	Eigen::VectorXd outputs_;
	Eigen::MatrixXd inputs_;

	Eigen::MatrixXd inputs_gradient_;

   public:
	virtual ~ALoss() = default;

	virtual void forward(const Eigen::MatrixXd& predictive_inputs,
						 const Eigen::VectorXi& target_inputs) = 0;
	virtual void backward(const Eigen::VectorXi& target_inputs) = 0;

	const Eigen::VectorXd& getOutputs() const;
	const Eigen::MatrixXd& getInputsGradient() const;
	double getLoss() const;

	static ALoss::lossFuncType getLossType(std::string_view str);

	friend std::ostream& operator<<(std::ostream& os, const ALoss& rhs);
};
