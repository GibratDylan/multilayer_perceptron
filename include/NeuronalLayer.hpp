#pragma once

#include <cstddef>
#include <ostream>

#include <Eigen/Dense>

class NeuronalLayer {
   private:
	Eigen::MatrixXd weights_;
	Eigen::VectorXd biases_;
	Eigen::MatrixXd outputs_;
	Eigen::MatrixXd inputs_;

	Eigen::MatrixXd weights_gradient_;
	Eigen::VectorXd biases_gradient_;
	Eigen::MatrixXd inputs_gradient_;

   public:
	explicit NeuronalLayer(size_t input_size, size_t num_neurons);
	NeuronalLayer(const NeuronalLayer& other) = default;
	NeuronalLayer& operator=(const NeuronalLayer& other) = default;
	NeuronalLayer(NeuronalLayer&& other) = default;
	NeuronalLayer& operator=(NeuronalLayer&& other) = default;

	void forward(const Eigen::MatrixXd& inputs);
	void backward(const Eigen::MatrixXd& inputs);
	Eigen::Index getInputSize() const;
	Eigen::Index getNumNeurons() const;
	const Eigen::MatrixXd& getOutputs() const;
	const Eigen::MatrixXd& getInputsGradient() const;

	friend std::ostream& operator<<(std::ostream& os, const NeuronalLayer& rhs);
};
