#pragma once

#include <Eigen/Dense>

class ALoss {
   protected:
	Eigen::VectorXd _outputs;
	virtual void forward(const Eigen::MatrixXd& predictive_inputs,
						 const Eigen::VectorXi& target_inputs) = 0;

   public:
	virtual ~ALoss() = default;
	double calculate(const Eigen::MatrixXd& predictive_inputs,
					 const Eigen::VectorXi& target_inputs);

	const Eigen::VectorXd& getOutputs() const;
};
