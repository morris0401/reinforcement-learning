import numpy as np
import argparse

def compute(args):
    rewards = np.array([np.load("./results_step3/baseline/BCQ_baseline_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))

    rewards = np.array([np.load("./results_step3/GAN/BCQ_GAN_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))

    rewards = np.array([np.load("./results_step3/quadruple/BCQ_quadruple_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))

    rewards = np.array([np.load("./results_step3/shared/BCQ_shared_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))

    rewards = np.array([np.load("./results_step3/no_perturbation/BCQ_no_perturbation_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))

    rewards = np.array([np.load("./results_step3/gamma09/BCQ_gamma09_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))

    '''rewards = np.array([np.load("./results_step3/batch200/BCQ_batch200_{}-v0_{}.npy".format(args.data, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    print(np.round(np.mean(rewards_avg[99:109]), 1))'''

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default='pen-human')
	args = parser.parse_args()
	compute(args)