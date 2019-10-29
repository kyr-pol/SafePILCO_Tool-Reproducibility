import numpy as np
import sys
from matplotlib import pyplot as plt
import seaborn as sns


#sns.set_context('paper')
sns.set(font_scale=1.5)
#plt.rcParams.update({'font.size': 20})
sns.set_style('white')

paths = ["results/mountain_car/", "results/inverted_pendulum/",
         "results/pend_swing_up/", "results/double_pendulum/", "results/swimmer/"]
names = ["mountain_car", "inverted_pendulum", "pend_swing_up", "double_pendulum", "swimmer"]
runs_so_far = [10,10,5,8,8] # this refers to the number of random seeds each experiment has been run on

for which_env in range(5):
    X = []
    X_eval =[]
    all_returns_sampled = []
    all_returns_full = []
    path = paths[which_env]
    name = names[which_env]
    for i in range(runs_so_far[which_env]):
        X.append(np.loadtxt(path + 'X_' + str(i) + '.csv', delimiter=','))
        X_eval.append(np.loadtxt(path + 'X_eval_'  + str(i) + '.csv', delimiter=','))
        all_returns_sampled.append(np.loadtxt(path + 'evaluation_returns_sampled_'  + str(i) + '.csv', delimiter=','))
        all_returns_full.append(np.loadtxt(path + 'evaluation_returns_full_'  + str(i) + '.csv', delimiter=','))

    # Lists = [seed, [iteration_number, evaluation_runs] 10 x 5 x 4
    evals_run = []
    for seed in all_returns_full:
        # average out evaluation runs
        evals_run.append(np.mean(seed, 1))

    evals_run = np.array(evals_run)
    # average out random seed
    means_per_iteration = np.mean(evals_run, 0)
    std_per_iteration = np.std(evals_run, 0)

    random_agents = np.loadtxt('random_policy.csv', delimiter=',')

    plt.plot(means_per_iteration, label='SafePILCO')
    plt.fill_between(range(len(means_per_iteration)), means_per_iteration + std_per_iteration, means_per_iteration - std_per_iteration, alpha=0.3)


    m = random_agents[0, which_env]
    s = random_agents[1, which_env]
    plt.plot(len(means_per_iteration)*[m], color='r', label='Random', linestyle='dashed')
    plt.fill_between(range(len(means_per_iteration)),m+s, m-s, color='r', alpha=0.3)
    #plt.legend()
    low = m-s
    high = max(means_per_iteration + std_per_iteration)
    r = high - low
    plt.ylim([low-0.1*r,high+0.1*r])
    plt.savefig("plots/" + name)
    plt.show()

    if which_env==4: # only for swimmer
        best_so_far = evals_run.copy()
        for seed in range(best_so_far.shape[0]):
            for iteration in range(1, best_so_far.shape[1]):
                if best_so_far[seed, iteration-1] < evals_run[seed, iteration]:
                    best_so_far[seed, iteration] = evals_run[seed, iteration]
                else:
                    best_so_far[seed, iteration] = best_so_far[seed, iteration-1]

        means_best_so_far = np.mean(best_so_far, 0)
        stds_best_so_far = np.std(best_so_far, 0)

        plt.plot(means_best_so_far, label='SafePILCO')
        plt.fill_between(range(len(means_best_so_far)), means_best_so_far + stds_best_so_far, means_best_so_far - stds_best_so_far, alpha=0.3)
        plt.savefig("plots/" + name + '_best_so_far')
        plt.show()
