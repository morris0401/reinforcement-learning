import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.xlabel('Time steps')
    plt.ylabel('Average Return')

def random(args):

    initialize_plot()
    plt.title('Hopper-random-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_hopper-random-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/hopper-random.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Walker2d-random-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_walker2d-random-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/walker2d-random.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Antmaze-medium-diverse-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_antmaze-medium-diverse-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/antmaze-medium-diverse.png".format(args.method))
    plt.show()
    plt.close()

def medium(args):

    initialize_plot()
    plt.title('Hopper-medium-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_hopper-medium-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/hopper-medium.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Walker2d-medium-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_walker2d-medium-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/walker2d-medium.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Antmaze-medium-play-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_antmaze-medium-play-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/antmaze-medium-play.png".format(args.method))
    plt.show()
    plt.close()

def expert(args):

    initialize_plot()
    plt.title('Hopper-expert-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_hopper-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/hopper-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Walker2d-expert-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_walker2d-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/walker2d-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Maze2d-umaze-v1')
    rewards = np.array([np.load("./results/{}/BCQ_{}_maze2d-umaze-v1_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/maze2d-umaze.png".format(args.method))
    plt.show()
    plt.close()

def medium_replay(args):

    initialize_plot()
    plt.title('Hopper-medium-replay-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_hopper-medium-replay-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/hopper-medium-replay.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Walker2d-medium-replay-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_walker2d-medium-replay-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/walker2d-medium-replay.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Maze2d-medium-v1')
    rewards = np.array([np.load("./results/{}/BCQ_{}_maze2d-medium-v1_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/maze2d-medium.png".format(args.method))
    plt.show()
    plt.close()

def medium_expert(args):

    initialize_plot()
    plt.title('Hopper-medium-expert-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_hopper-medium-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/hopper-medium-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Walker2d-medium-expert-v0')
    rewards = np.array([np.load("./results/{}/BCQ_{}_walker2d-medium-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/walker2d-medium-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Maze2d-large-v1')
    rewards = np.array([np.load("./results/{}/BCQ_{}_maze2d-large-v1_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots/{}/maze2d-large.png".format(args.method))
    plt.show()
    plt.close()

if __name__ == "__main__":

    os.makedirs("./Plots", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='baseline')
    args = parser.parse_args()

    random(args)
    medium(args)
    expert(args)
    medium_replay(args)
    medium_expert(args)