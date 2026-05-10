#pragma once

#include <Eigen/Dense>

class AActivation {
   protected:
	Eigen::MatrixXd _outputs;

   public:
	virtual void forward(const Eigen::MatrixXd& inputs) = 0;
	virtual ~AActivation() = default;

	const Eigen::MatrixXd& getOutputs() const;
};
