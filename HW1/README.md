# HW1

This directory contains the files for **HW1**.

## 📄 Project Introduction

# Foundational Reinforcement Learning: Theory, Algorithms, and Offline Data Analysis

## Introduction
Reinforcement Learning (RL) presents significant challenges in optimizing sequential decision-making, necessitating a robust understanding of its theoretical underpinnings and practical algorithms. This project delves into the core principles of RL, spanning fundamental theoretical proofs, the design and implementation of classical dynamic programming algorithms, and the practical exploration of real-world offline RL datasets. Our aim is to synthesize foundational concepts—from Bellman optimality and contraction mappings to policy gradient mechanics—with hands-on experience, bridging the gap between mathematical theory and its application in contemporary RL research, particularly within the offline learning paradigm.

## Methods
Our approach integrates theoretical derivations with algorithmic design and practical data analysis.

### Theoretical Foundations
We established the mathematical bedrock of optimal control by proving the Bellman optimality equations for optimal state-value and action-value functions $V^*(s), Q^*(s,a)$. We rigorously demonstrated that the Bellman optimality operator ($T^*$) for Q-Value Iteration is a $\gamma$-contraction in the $L_\infty$ norm, a crucial property guaranteeing convergence. This was extended to regularized Markov Decision Processes (MDPs), verifying that the regularized Bellman expectation operator ($T_\pi^\Omega$) also exhibits $\gamma$-contraction in $L_\infty$. Furthermore, we derived a key identity relating expected discounted rewards over trajectories to discounted state-action visitations, providing theoretical insight essential for policy gradient methods.

### Algorithmic Design and Implementation
Building on theoretical insights, we designed an iterative Value Iteration-like algorithm to compute optimal value functions ($V_\Omega^*$, $Q_\Omega^*$) for regularized MDPs, incorporating regularization terms like Shannon entropy. For classic MDPs, we implemented Policy Iteration and Value Iteration algorithms, solving the OpenAI Gym "Taxi-v3" environment with a discount factor $\gamma=0.9$ and a convergence criterion $\epsilon=10^{-3}$. These implementations serve to validate the theoretical convergence properties in a practical setting.

### Offline Reinforcement Learning with D4RL
To explore practical challenges in contemporary RL, we engaged with the D4RL (Datasets for Reinforcement Learning) library. This involved setting up the environment, loading, and systematically inspecting large-scale offline datasets from diverse domains. Specifically, we analyzed `maze2d-umaze-v1` and `walker2d-medium-v2`, characterizing their observation, action, and reward structures. Our investigation included sampling random actions within these environments to observe immediate feedback and understand the inherent challenges of learning from pre-collected, potentially suboptimal data.

## Results
The project yielded several key outcomes across its theoretical and practical components.

### Theoretical Validations
Our proofs confirmed the fundamental Bellman optimality conditions and the critical $\gamma$-contraction property of both standard and regularized Bellman operators, mathematically ensuring the convergence of iterative solution methods. The derivation of the policy gradient property elucidated the connection between state-action visitations and expected returns, a cornerstone for gradient-based policy optimization.

### Algorithmic Performance
The implementations of Policy Iteration and Value Iteration for the "Taxi-v3" environment successfully converged, demonstrating alignment in their resulting optimal policies with zero discrepancy, affirming the correctness and robustness of these dynamic programming approaches.

### D4RL Dataset Insights
Analysis of the D4RL datasets revealed distinct characteristics. The `maze2d-umaze-v1` dataset (1,000,000 samples) showed 4-element observations and 2-element actions, with rewards consistently zero and episodes rarely terminating under random exploration. This highlighted the sparse reward challenge inherent in navigating such environments with naive policies. In contrast, the `walker2d-medium-v2` dataset (1,000,000 samples) presented a more complex 17-element observation and 6-element action space, exhibiting varied, non-zero rewards that generally decreased under random policies. These observations underscore the difficulty of learning effective policies from arbitrary trajectories and emphasize the need for advanced offline RL techniques to leverage such data.

## Conclusion
Through this project, we achieved a comprehensive exploration of Reinforcement Learning fundamentals, integrating theoretical rigor with practical algorithmic implementation and real-world data analysis. By establishing firm theoretical understanding of MDPs, demonstrating the efficacy of core RL algorithms, and gaining hands-on experience with the complexities of offline datasets, we lay a solid foundation for addressing advanced challenges in RL research. The insights gained from characterizing D4RL environments, particularly regarding reward sparsity and suboptimality of exploratory data, underscore the critical importance of robust offline RL methodologies for developing capable agents in complex environments.

## 📂 Files

- `111550177_RL_HW1.pdf`
- `Spring2024_RL_HW1.pdf`

A detailed project description and summary can be found in the [main repository README](../README.md).

