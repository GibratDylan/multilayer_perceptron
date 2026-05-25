#pragma once

#include <ostream>

#include <Eigen/Dense>

class ALoss {
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

	friend std::ostream& operator<<(std::ostream& os, const ALoss& rhs);
};
