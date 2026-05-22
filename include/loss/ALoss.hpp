#pragma once

#include <ostream>

#include <Eigen/Dense>

class ALoss {
   protected:
	Eigen::VectorXd outputs_;
	virtual void forward(const Eigen::MatrixXd& predictive_inputs,
						 const Eigen::VectorXi& target_inputs) = 0;

   public:
	virtual ~ALoss() = default;
	double calculate(const Eigen::MatrixXd& predictive_inputs,
					 const Eigen::VectorXi& target_inputs);

	const Eigen::VectorXd& getOutputs() const;
};

std::ostream& operator<<(std::ostream& os, const ALoss& rhs);
