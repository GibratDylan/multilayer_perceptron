#include <algorithm>
#include <array>
#include <execution>
#include <functional>
#include <iostream>
#include <numeric>

int main() {
  const std::array<float, 4> inputs{1.f, 2.f, 3.f, 2.5f};
  const std::array<std::array<float, 4>, 3> weights{
      {{0.2f, 0.8f, -0.5f, 1.0f},
       {0.5f, -0.91f, 0.26f, -0.5f},
       {-0.26f, -0.27f, 0.17f, 0.87f}}};
  const std::array<float, 3> biases{2.0f, 3.0f, 0.5f};

  std::array<float, 3> outputs{};

  std::transform(weights.cbegin(), weights.cend(), biases.cbegin(),
                 outputs.begin(),
                 [&inputs](const auto &weight, const float bias) {
                   return std::transform_reduce(
                       std::execution::par, inputs.cbegin(), inputs.cend(),
                       weight.cbegin(), bias, std::plus<float>(),
                       std::multiplies<float>());
                 });

  std::cout << '[';
  std::for_each(outputs.cbegin(), outputs.cend(),
                [](auto &ite) { std::cout << ite << ", "; });
  std::cout << "]\n";

  return 0;
}