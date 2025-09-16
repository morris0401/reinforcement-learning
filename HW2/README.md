# HW2

This directory contains the files for **HW2**.

## 📄 Project Introduction

# Toward Stable and Efficient Policy Optimization in Deep Reinforcement Learning

## Introduction

Policy gradient methods are foundational in reinforcement learning (RL), enabling agents to learn complex behaviors without requiring an explicit model of the environment. However, a significant challenge in applying these methods, especially with deep neural network function approximators, is the high variance in gradient estimates, which can lead to unstable training and slow convergence. Furthermore, understanding the theoretical convergence properties of these algorithms in non-convex settings remains an active area of research. This project addresses these critical issues by integrating theoretical analysis of policy optimization with rigorous empirical investigation into variance reduction techniques and neural network configurations. Our goal is to enhance the stability, efficiency, and theoretical understanding of policy gradient algorithms, paving the way for more robust deep RL applications.

## Methods

Our approach combines theoretical derivations with practical implementation and experimentation across standard RL benchmarks.

### Theoretical Foundations
We delve into the mathematical underpinnings of policy gradient methods and their convergence properties:
*   **Policy Gradient Variance Reduction:** We derive the mean vector and covariance matrix of policy gradients, both with and without a value function baseline, for softmax policies. A key contribution is the derivation of an optimal state-dependent baseline $B(s)$ that explicitly minimizes the trace of the policy gradient's covariance matrix, providing a theoretical basis for variance reduction.
*   **Polyak-Lojasiewicz (PL) Condition:** We prove a non-uniform Polyak-Lojasiewicz condition for policy optimization under softmax policies. This proof leverages the Cauchy-Schwarz inequality, the policy gradient expression, and the Performance Difference Lemma, providing theoretical insights into the local convergence guarantees of policy gradient methods even in non-convex landscapes.
*   **Monte Carlo Policy Evaluation:** We analyze the bias inherent in every-visit Monte Carlo estimates within a simple 2-state Markov Reward Process, deriving the true value function and the expected value of the Monte Carlo estimate to quantify this bias.

### Algorithmic Implementation and Experimentation
We implemented and evaluated three core policy gradient algorithms using neural network function approximators in OpenAI Gym environments:
*   **Vanilla REINFORCE:** A baseline implementation for "CartPole-v0".
*   **REINFORCE with Value Function Baseline:** An extension incorporating a learned value function as a baseline for variance reduction, tested on "LunarLander-v2".
*   **REINFORCE with Generalized Advantage Estimation (GAE):** An advanced technique utilizing GAE for improved advantage estimation, also applied to "LunarLander-v2".

**Neural Network Architecture:** All experimental agents utilized Multilayer Perceptrons (MLPs) with two shared hidden layers and Rectified Linear Unit (ReLU) activation functions. Dropout (0.2 probability) was employed in initial experiments. A value function was consistently used as a baseline, with GAE explored as an alternative advantage estimation method.

**Experimental Setup:** We conducted systematic hyperparameter tuning for hidden layer size (128-256), discount factor ($\gamma = 0.999$), learning rates (0.01, 0.02), and the GAE parameter ($\lambda$). Training progress, Exponentially Weighted Moving Average (EWMA) reward, episode reward, and convergence speed were meticulously monitored using TensorBoard.

## Results

Our investigations yielded significant theoretical and empirical insights into the performance and stability of policy gradient methods:

*   **Theoretical Contributions:**
    *   The derivation of an optimal state-dependent baseline formally demonstrates how variance in policy gradient estimates can be theoretically minimized.
    *   The proven non-uniform Polyak-Lojasiewicz condition provides a stronger theoretical guarantee for local convergence than typical smoothness assumptions, offering a deeper understanding of policy gradient optimization landscapes.
    *   The analysis of Monte Carlo bias quantifies the estimation inaccuracies, informing robust evaluation strategies.

*   **Empirical Performance and Stability:**
    *   **Convergence Sensitivity:** We observed that higher learning rates consistently inhibited model convergence. Conversely, a decrease in the discount factor ($\gamma$) generally led to an increase in the number of episodes required for convergence, underscoring the critical role of these hyperparameters. Normalization of return values was identified as paramount for successful model convergence.
    *   **Effectiveness of Baselines:** Directly utilizing a learned value function as a baseline significantly improved performance and stability compared to custom or no baselines.
    *   **GAE Impact:** Generalized Advantage Estimation (GAE) demonstrably improved reward stability and final performance. Experiments showed a clear relationship between the GAE $\lambda$ parameter and convergence speed: a larger $\lambda$ (e.g., 0.999) led to significantly faster convergence (991 episodes), outperforming smaller values like 0.8 (1121 episodes) or 0.7 (2476 episodes). Very small $\lambda$ values (e.g., 0.5) severely impeded convergence, often failing within 10,000 episodes.
    *   **Optimal Hyperparameters:** For the environments tested, an optimal hidden layer size range was found to be 128-256.

In conclusion, this project provides a comprehensive theoretical and empirical investigation into the challenges and solutions for stable and efficient policy optimization in deep reinforcement learning. Our findings highlight the profound impact of variance reduction techniques like GAE and well-tuned hyperparameters on algorithm convergence and overall performance, while our theoretical proofs contribute to a deeper understanding of policy gradient optimization dynamics.

## 📂 Files

- `111550177_RL_HW2.pdf`
- `Spring2024_RL_HW2_updated.pdf`

A detailed project description and summary can be found in the [main repository README](../README.md).

