# Optimization Algorithms for Training Neural Networks 

This repository implements multiple optimization algorithms from scratch to train a deep neural network on a 2D classification dataset. It includes:

- Batch Gradient Descent (GD)
- Mini-batch Momentum
- Adam Optimization

All components are built using NumPy and Matplotlib without high-level machine learning libraries.

---

## 🚀 Key Features

- 📉 Cost tracking and plotting across training epochs
- 🧮 Mini-batch creation for stable gradient updates
- 📦 Modular optimizer support: GD, Momentum, Adam
- 📈 Decision boundary visualization per optimizer
- 🔍 Forward and Backward Propagation logic implemented from scratch

---

## 🧠 Network Architecture

A fully connected neural network with the following layer configuration:

```text
Input Layer (n_x) → Hidden Layer 1 (5 neurons) → Hidden Layer 2 (2 neurons) → Output Layer (1 neuron)
Activation functions: ReLU and Sigmoid

Loss function: Binary Cross Entropy

Optimizers: GD, Momentum, Adam (modularly selectable)

🛠️ Optimization Algorithms Implemented
1. Gradient Descent (GD)
Basic update rule:

go
Copy
Edit
W := W - α * dW
b := b - α * db
2. Momentum
Uses exponentially weighted moving averages of gradients for faster convergence.

go
Copy
Edit
v := β * v + (1 - β) * dθ
θ := θ - α * v
3. Adam
Combines Momentum and RMSprop:

Maintains both momentum (v) and squared gradients (s)

Applies bias correction

🔁 Training Pipeline
Mini-batch Sampling

Randomly partitions data into mini-batches

Ensures stability in training

Forward Propagation

Applies linear transformations and activations

Cost Computation

Cross-entropy loss

Backward Propagation

Uses cached values from forward pass to compute gradients

Parameter Update

Uses selected optimizer to update weights and biases

Cost Plotting

Tracks cost vs epochs for performance comparison

📊 Results & Visualization
For each optimizer, the decision boundary is plotted showing classification performance.

🟩 Gradient Descent: Baseline convergence

⚡ Momentum: Accelerated training with fewer oscillations

🔮 Adam: Most stable and adaptive

📦 File Overview
opt_utils_v1a.py: Utility functions for forward/backward propagation, cost, prediction, and plotting

testCases.py: Validation utilities for unit tests

main.ipynb or train.py: Main script containing the training loop and optimizer comparison

🧪 Dependencies
Install with:

bash
Copy
Edit
pip install numpy matplotlib scikit-learn
🖼️ Sample Output
Each optimization method produces a plot similar to:


Above: Decision boundary for the trained model with Adam optimizer.

