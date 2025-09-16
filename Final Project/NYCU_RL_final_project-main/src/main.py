import argparse
import gym
import d4rl
import numpy as np
import os
import torch

import BCQ
import utils



# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args, seed):
	# For saving files
	setting = f"{args.env}_{seed}"

	# Initialize policy
	policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

	# Load buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	dataset = d4rl.qlearning_dataset(env)
	N = dataset['rewards'].shape[0]
	for i in range(N):
		obs = dataset['observations'][i]
		new_obs = dataset['next_observations'][i]
		action = dataset['actions'][i]
		reward = dataset['rewards'][i]
		done_bool = bool(dataset['terminals'][i])
		replay_buffer.add(obs, action, new_obs, reward, done_bool)
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	
	while training_iters < args.max_timesteps: 
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

		evaluations.append(eval_policy(policy, args.env, seed))
		np.save(f"./results_step3/batch200/BCQ_batch200_{setting}", evaluations)

		training_iters += args.eval_freq
		print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default='relocate-expert-v0')        # OpenAI gym environment name
	#parser.add_argument("--seed", default=2, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=float)     # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
	parser.add_argument("--batch_size", default=200, type=int)      # Mini batch size for networks
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
	parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
	args = parser.parse_args()

	for seed in range(1, 3):
		print("---------------------------------------")	
		print(f"Setting: Training BCQ, Env: {args.env}, Seed: {seed}")
		print("---------------------------------------")

		if not os.path.exists("./results_step3"):
			os.makedirs("./results_step3")
		
		env = gym.make(args.env)

		env.seed(seed)
		env.action_space.seed(seed)
		torch.manual_seed(seed)
		np.random.seed(seed)
		
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		train_BCQ(state_dim, action_dim, max_action, device, args, seed)