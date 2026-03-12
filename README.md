# Hardware-Accelerated Neural Network — Software (PS)
MNIST digit recognition neural network running on the software side (PS) of the AMD AUP-ZU3 board. The goal is to write and time a software baseline, then slowly move the heavy matrix multiplication layers to the FPGA (PL) and measure the speedup.

**Team:** Ethan Ostroff & Neirit Mahabub

---

## Project Overview
The neural network classifies handwritten digits (0–9) from the MNIST dataset using a 3-layer fully connected architecture (input layer + 2 learned layers):
```
Input Layer (784) → Hidden Layer (128) + ReLU → Output Layer (10) + Softmax → Predicted Digit
```

| Layer | Role | Neurons | Notes |
|---|---|---|---|
| Input | Raw pixel data | 784 | 28×28 grayscale image, flattened |
| Hidden | Processing | 128 | Dense + ReLU activation |
| Output | Class scores | 10 | Dense + Softmax → digit 0–9 |

The transition from Input→Hidden (`784 → 128`) is the primary FPGA acceleration target since it dominates the multiply/accumulate operation count. The Hidden→Output transition (`128 → 10`) and softmax can remain on the processor if needed.

---

## Repo Structure
```
neural-network-software/
├── board_files/        # Inference notebook (run on AUP-ZU3 board)
│   └── board_inference.ipynb
│       - NumPy forward pass (no PyTorch/Keras needed)
│       - Loads .npy weight files
│       - Times Layer 1 (784→128) and Layer 2 (128→10)
│
└── training/           # Training notebook (run on laptop)
    └── neural_network.ipynb
        - Trains the model using Keras/TensorFlow
        - Exports weights as w1.npy, b1.npy, w2.npy, b2.npy
```

---

## Dependencies
| Environment | Libraries |
|---|---|
| Laptop (training) | `tensorflow`, `keras`, `numpy`, `matplotlib` |
| AUP-ZU3 board (inference) | `numpy`, `matplotlib` |
