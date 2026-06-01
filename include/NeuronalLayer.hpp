#pragma once

#include "types/eigen_types.hpp"

#include <ostream>

class NeuronalLayer {
   private:
	Matrix weights_;
	Vector biases_;
	Matrix outputs_;
	Matrix inputs_;

	Matrix weights_gradient_;
	Vector biases_gradient_;
	Matrix inputs_gradient_;

   public:
	explicit NeuronalLayer(int64_t input_size, int64_t num_neurons);

	void Forward(MatrixIn input_batch);
	void Backward(MatrixIn gradient_batch);
	int64_t GetInputSize() const;
	int64_t GetNumNeurons() const;
	MatrixIn GetOutputs() const;
	MatrixIn GetInputsGradient() const;

	friend std::ostream& operator<<(std::ostream& os, const NeuronalLayer& rhs);
};
