# HW3

This directory contains the files for **HW3**.

## 📄 Project Introduction

# Advancing Continuous Control with Trust Region and Policy Optimization Methods

## Introduction
Achieving robust and efficient control in environments with continuous action spaces remains a fundamental challenge in Reinforcement Learning (RL). Policy gradient methods, especially those incorporating stability mechanisms, have shown significant promise. This work explores the theoretical underpinnings and practical applications of advanced policy optimization algorithms, specifically Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), and Deep Deterministic Policy Gradient (DDPG). Our objective is to deepen the understanding of their operational principles, including their constrained optimization landscapes and update mechanisms, and to demonstrate their efficacy in solving complex continuous control tasks through deep neural network implementations.

## Methods
Our investigation spanned both theoretical analysis and practical implementation of key policy gradient algorithms.

### Theoretical Foundations of Policy Optimization
The theoretical component focused on the stability and convergence properties of TRPO and PPO. For TRPO, we rigorously proved two critical properties of its surrogate function `L_pi_old(pi_theta)`, showing `L_pi_old(pi_old) = eta(pi_old)` and the equality of their gradients at `theta=theta_old`. Furthermore, we applied Lagrangian duality to solve an approximated TRPO optimization problem, deriving the dual function `D(lambda)` and subsequently determining the optimal `lambda*` and policy parameter update `theta*`.

For PPO, we conducted a comparative analysis of its two clipped objective function variants: `L_clip(theta; theta_k)` and `L_clip_tilde(theta; theta_k)`. This comparison highlighted their distinct clipping behaviors on the probability ratio and advantage function, illustrating how these differences influence policy update gradients and overall objective values.

### Deep Deterministic Policy Gradient (DDPG) for Continuous Control
The practical component involved implementing the DDPG algorithm, a model-free, off-policy algorithm designed for continuous action spaces. DDPG employs an Actor-Critic architecture, utilizing deep neural networks for both policy (Actor) and value (Critic) approximations, alongside an experience replay buffer and target networks for stable learning.

We applied DDPG to two OpenAI Gym environments: "Pendulum-v0" (a 1-dimensional continuous control task) and "HalfCheetah" (a high-dimensional, continuous locomotion task using MuJoCo).

**Model Architectures:**
*   **Actor Networks:** Utilized linear layers with ReLU activations, culminating in a Tanh activation for the output layer to constrain actions. Output ranges were normalized (e.g., [-2, 6]).
    *   *Pendulum-v0:* Three linear layers, hidden size 100.
    *   *HalfCheetah:* Five linear layers, hidden size 128.
*   **Critic Networks:** Employed linear layers with ReLU activations, outputting a single value.
    *   *Pendulum-v0:* Three linear layers, hidden size 100.
    *   *HalfCheetah:* Five linear layers, hidden size 128.

**Hyperparameters (Selected):**
*   **Pendulum-v0:** `num_episodes=300`, `gamma=0.995`, `tau=0.002`, `noise_scale=0.3`, `replay_size=100000`, `batch_size=256`.
*   **HalfCheetah:** `num_episodes=200` (out of 300 planned), `gamma=0.995`, `tau=0.0005`, `lr_a=0.002`, `lr_c=0.006`, `noise_scale=0.3`, `replay_size=100000`, `batch_size=512`.

Training progress was monitored using Tensorboard, tracking actor loss, critic loss, and Exponentially Weighted Moving Average (EWMA) rewards.

## Results
The DDPG implementations yielded distinct performance profiles across the two environments.

For **Pendulum-v0**, the DDPG agent successfully learned a stable policy over 300 episodes. The EWMA reward converged to a range of -270 to -300, indicating a well-performing agent capable of balancing the pendulum. Actor and Critic loss plots exhibited typical decreasing trends, signifying successful convergence of the network parameters.

In the more challenging **HalfCheetah** environment, training for 200 episodes showed promising initial performance. The agent achieved rewards exceeding 1000 around episode 100, with the EWMA reward peaking at approximately 547. However, this promising trend was not sustained; performance subsequently declined, with rewards rapidly dropping to around -2400 in later episodes. This suggests that while DDPG demonstrated the capacity for initial learning, achieving robust and high-performing policies for complex locomotion tasks like HalfCheetah often requires more extensive hyperparameter tuning and potentially longer training durations. Future work will focus on systematic hyperparameter optimization to attain target performance benchmarks (e.g., 4000+ score).

## 📂 Files

- `111550177_HW3.pdf`
- `Spring2024_RL_HW3.pdf`

A detailed project description and summary can be found in the [main repository README](../README.md).

