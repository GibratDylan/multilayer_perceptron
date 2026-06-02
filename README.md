# Multilayer Perceptron — from scratch in C++ / Eigen

A feed-forward neural network (multilayer perceptron) implemented from scratch in
modern C++, using **Eigen** for linear algebra and **no machine-learning framework**.
The focus is not on reaching a framework's accuracy — it is on building the engine
itself: forward/backward propagation, training loop, and a numerically careful,
allocation-conscious data path.

> **Status:** in progress.

---

## Design highlights

These are the engineering decisions the codebase is built around:

- **From scratch, framework-free.** Forward pass, backpropagation, and gradient
  updates are implemented by hand; Eigen is used only as a dense linear-algebra
  backend.
- **Column-major, one-sample-per-column convention.** Batches are stored with each
  sample as a column, which keeps the hot matrix operations contiguous in Eigen's
  column-major memory and turns a training step into a few large GEMM calls instead
  of many small ones.
- **Zero-copy batching with `Eigen::Ref`.** Mini-batches are passed as
  non-owning `Eigen::Ref` views into the dataset, so iterating over batches does
  not allocate or copy per step.
- **Pluggable components via the Strategy pattern.** Activation functions and
  optimizers are interchangeable behind a common interface, so a new activation or
  optimizer can be added without touching the training loop.
- **Explicit precision trade-offs.** The numeric type is configurable (FP32 / FP64),
  trading memory footprint and speed against numerical headroom.
- **Allocation-conscious data layer.** Buffers are sized up front and reused across
  the training loop to keep per-epoch heap traffic low.

---

## Architecture

| Module        | Responsibility                                                        |
|---------------|-----------------------------------------------------------------------|
| `Network`     | Layer stack, forward/backward pass, weight updates                    |
| `Layer`       | Dense layer: weights, biases, cached activations for backprop         |
| `Activation`  | Strategy interface + implementations (e.g. ReLU, sigmoid, softmax)    |
| `Optimizer`   | Strategy interface + implementations (e.g. SGD, momentum)             |
| `Loss`        | Loss functions and their gradients                                    |
| `Data`        | Dataset loading, normalization, batch iteration (`Eigen::Ref` views)  |

---

## Build

**Requirements:** a C++ compiler <!-- TODO: state the standard, e.g. C++17 --> and
the Eigen headers.

```bash
# if CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# if Makefile
make
```

---

## Usage

```bash
# Train a model and save the learned weights
./mlp train --data <path> --epochs <n> --lr <rate>

# Run inference / evaluate on held-out data
./mlp predict --data <path> --model <weights>
```

**Results:** <!-- TODO: one concrete number a reader can latch onto, e.g. final
loss / accuracy on the validation split. Even a single honest figure beats none. -->

---

## Roadmap

- [x] Forward / backward propagation
- [ ] Mini-batch training loop with zero-copy batching
- [ ] Additional optimizers

---

## Code quality

The codebase follows the **Google C++ Style Guide**, enforced with `.clang-tidy`
and `.clang-format` config checked into the repo.

```bash
clang-format -i <files>     # formatting
clang-tidy <files>          # static analysis
```

---

## References

- [Eigen documentation](https://eigen.tuxfamily.org/dox/)
- Neural Networks and Deep Learning — Michael Nielsen (the from-scratch backprop reference)
