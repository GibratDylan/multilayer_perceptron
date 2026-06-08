#pragma once

#include "types/eigen_types.hpp"

#include <cstdint>
#include <ostream>

class NeuronalLayer {
   private:
	Matrix weights_{};
	Vector biases_{};
	Matrix outputs_{};
	Matrix inputs_{};

	Matrix weights_gradient_{};
	Vector biases_gradient_{};
	Matrix inputs_gradient_{};

   public:
	explicit NeuronalLayer(int64_t input_size, int64_t num_neurons);

	void Forward(const MatrixIn& input_batch);
	void Backward(const MatrixIn& gradient_batch);

	int64_t GetInputSize() const noexcept;
	int64_t GetNumNeurons() const noexcept;
	MatrixIn GetOutputs() const noexcept;
	MatrixIn GetInputsGradient() const noexcept;
	MatrixIn GetWeightsGradient() const noexcept;
	MatrixIn GetBiasesGradient() const noexcept;

	friend std::ostream& operator<<(std::ostream& os, const NeuronalLayer& rhs);
};
