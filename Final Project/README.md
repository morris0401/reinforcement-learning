# Final Project

This directory contains the files for **Final Project**.

## 📄 Project Introduction

# Offline Reinforcement Learning with Modified Batch-Constrained Q-learning

## Introduction
Offline Reinforcement Learning (RL) presents a compelling paradigm for training intelligent agents from pre-collected datasets without costly and potentially unsafe environmental interactions. A fundamental challenge in this setting is the "extrapolation error" or out-of-distribution (OOD) state-action pairs. When a learned policy deviates from the data distribution it was trained on, Q-value estimations become unreliable, leading to suboptimal or catastrophic behavior. Batch-Constrained Q-learning (BCQ) addresses this by explicitly restricting the learned policy to actions likely to be found within the dataset. BCQ achieves this through a combination of a Variational Auto-Encoder (VAE) to generate in-batch actions, a perturbation model to diversify these actions, and Clipped Double Q-learning for robust value estimation. This project conducts an extensive ablation study on BCQ, exploring six distinct architectural and hyperparameter modifications to enhance its performance and better mitigate the inherent challenges of OOD state-action pairs across diverse D4RL benchmarks.

## Methods
Our investigation establishes a DDPG-based BCQ baseline, adapted to utilize standard D4RL datasets. The baseline architecture features a VAE with 750-node hidden layers, an Actor (perturbation model) with 400 and 300-node hidden layers, and two Critics (Clipped Double Q-learning) each with 400 and 300-node hidden layers. Training employs Adam optimizers, a batch size of 100, discount factor (Gamma) of 0.99, and specific learning rates (0.001) for Actor, Critic, and VAE.

We systematically explored six modifications to this baseline:
1.  **Conditional Generative Adversarial Network (CGAN) for VAE Replacement:** To evaluate if an alternative generative model, trained to differentiate real from generated actions, could more effectively model the data distribution and enhance exploration. The CGAN architecture consists of 3-layer Generator and Discriminator networks, utilizing Leaky ReLU and Tanh (Generator) or Sigmoid (Discriminator) activations.
2.  **Clipped Quadruple Q-learning:** Extending the standard Clipped Double Q-learning to four Q-networks, aiming for more consistent and robust value estimation by minimizing over a larger set of value estimates. Critic hidden layer sizes were adjusted to (200, 150) due to computational constraints.
3.  **Shared First Layer for Actor and Critic:** Introducing a shared initial neural network layer between the Actor and Critic to reduce model complexity, parameter count, and computational overhead, while hypothesizing shared feature learning could be beneficial.
4.  **Removal of the Perturbation Model:** Eliminating the actor's perturbation component, forcing the policy to rely solely on actions generated directly by the VAE and selected by the Q-networks. This aims to severely restrict actions to the observed data distribution, prioritizing safety over exploration.
5.  **Modified Discount Factor (Gamma = 0.9):** Investigating the impact of a shorter planning horizon on policy learning and long-term reward considerations.
6.  **Increased Batch Size (200):** Examining whether a larger sample size during training updates can stabilize learning and improve policy generalization.

Each modification was trained for 1 million steps on various D4RL datasets (e.g., `hopper`, `walker2d`, `maze2d`, `antmaze`, `pen`, `hammer`, `door`, `relocate`) over three random seeds. Policy performance was evaluated every 5000 steps, with final results averaged over 10 evaluations within the 500k to 545k training step range and then across the three seeds.

## Results
Our extensive empirical evaluation revealed that no single modification consistently outperformed the original BCQ across all D4RL tasks, highlighting a complex interplay between architectural choices and dataset characteristics.

*   **CGAN for VAE Replacement:** This modification showed improved or comparable performance primarily in exploration-heavy environments like `random` and `maze` datasets (e.g., `Hopper-random`, `Maze2d-large`), where its generative capacity might facilitate beneficial exploration. However, it performed poorly in `expert` or complex Adroit environments (e.g., `Relocate-expert`), where precise action control is critical, often leading to oscillating training curves.
*   **Quadruple Q-learning:** Generally maintained performance near the baseline but demonstrated significant improvements in `maze` environments (e.g., `Maze2d-umaze`), `hopper-expert`, and `walker2d-expert`. This suggests that using four Q-networks can lead to more consistent reward estimations, particularly in environments with unknown states or when striving for more robust value predictions.
*   **Shared First Layer:** Consistently led to worse performance across most tasks, particularly in `expert` environments (e.g., `Hopper-expert`). This reduction in model complexity appears to hinder the model's ability to learn intricate mappings required for complex tasks, or the shared layer did not provide sufficiently discriminative common features for both actor and critic.
*   **Removal of Perturbation Model:** Delivered mixed results. It often performed worse in environments with diverse or random data distributions (e.g., `Walker2d-medium-replay`) due to a lack of exploratory capacity. Crucially, it showed notable improvements in `expert` and complex Adroit tasks (e.g., `Hopper-expert`, `Relocate-expert`, `Hammer-human`). This indicates that in expert settings, restricting the policy to actions highly consistent with the observed (expert) data via the VAE alone can be safer and more effective, preventing divergence into OOD regions.
*   **Modified Discount Factor (Gamma = 0.9):** Generally resulted in significantly weaker performance compared to the baseline's 0.99, indicating that a longer planning horizon is typically more suitable. Surprisingly, it showed better performance in specific complex `expert` tasks like `Hammer-expert` and `Door-human`, suggesting task-dependent optimal discount factors.
*   **Increased Batch Size (200):** The effect of a larger batch size was highly task-dependent. While it led to substantial improvements in `Maze2d-large`, it deteriorated performance in others, such as `Walker2d-medium-replay`. This highlights that optimal batch size is not universal and can vary based on dataset characteristics and task complexity.

Overall, our study underscores the inherent trade-offs in offline RL: modifications that enhance exploration (e.g., CGAN in random tasks) can be detrimental in expert settings where adherence to known trajectories is paramount. Conversely, restricting policy exploration (e.g., removing perturbation) can be beneficial in expert data but harmful in more diverse datasets. These findings emphasize that BCQ, despite its mechanisms, still struggles with consistently handling OOD state-action pairs across diverse environments. Future work could focus on hybrid approaches that adapt the level of exploration based on dataset characteristics, and more robust algorithms that intrinsically manage OOD data.

## 📂 Files

- `RL_final_project_technical report_compressed.pdf`
- `Team11_Slides_compressed.pdf`

A detailed project description and summary can be found in the [main repository README](../README.md).

