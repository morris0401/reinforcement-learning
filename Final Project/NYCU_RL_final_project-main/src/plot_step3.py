import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def initialize_plot():
    plt.figure(figsize=(10, 5))
    plt.xlabel('Time steps')
    plt.ylabel('Average Return')

def human(args):

    initialize_plot()
    plt.title('Pen-human-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_pen-human-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/pen-human.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Hammer-human-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_hammer-human-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/hammer-human.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Door-human-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_door-human-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/door-human.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Relocate-human-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_relocate-human-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/relocate-human.png".format(args.method))
    plt.show()
    plt.close()

def cloned(args):

    initialize_plot()
    plt.title('Pen-cloned-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_pen-cloned-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/pen-cloned.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Hammer-cloned-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_hammer-cloned-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/hammer-cloned.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Door-cloned-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_door-cloned-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/door-cloned.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Relocate-cloned-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_relocate-cloned-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/relocate-cloned.png".format(args.method))
    plt.show()
    plt.close()

def expert(args):

    initialize_plot()
    plt.title('Pen-expert-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_pen-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/pen-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Hammer-expert-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_hammer-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/hammer-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Door-expert-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_door-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/door-expert.png".format(args.method))
    plt.show()
    plt.close()

    initialize_plot()
    plt.title('Relocate-expert-v0')
    rewards = np.array([np.load("./results_step3/{}/BCQ_{}_relocate-expert-v0_{}.npy".format(args.method, args.method, seed)) for seed in range(3)]).transpose()
    rewards_avg = np.mean(rewards, axis = 1)
    rewards_std = np.std(rewards, axis = 1)
    rewards_avg = np.insert(rewards_avg, 0, 0)
    rewards_std = np.insert(rewards_std, 0, 0)

    x = np.linspace(0, 1e6, 201)
    plt.plot(x, rewards_avg, color = 'orange')
    plt.fill_between(x, rewards_avg + rewards_std, rewards_avg - rewards_std, facecolor = 'lightblue')
    plt.savefig("./Plots_step3/{}/relocate-expert.png".format(args.method))
    plt.show()
    plt.close()

if __name__ == "__main__":

    os.makedirs("./Plots_step3", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='baseline')
    args = parser.parse_args()

    human(args)
    cloned(args)
    expert(args)