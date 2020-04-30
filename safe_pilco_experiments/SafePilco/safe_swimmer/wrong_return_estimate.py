import numpy as np
from matplotlib import pyplot as plt
import sys

import seaborn as sns

#sns.set_context('paper')
sns.set(font_scale=1.5)
#plt.rcParams.update({'font.size': 20})
sns.set_style('white')

# names = ["old_new/gym_swimmer", "old_new/old_swimmer", "old_new/ld_swimmer_highBf", "old_new/gym_swimmer_highBf",        #4
# "old_new/gym_swimmer_gymRet", "old_new/old_swimmer_gymRet", "old_new/gym_swimmer_noSubs", "old_new/old_swimmer_noSubs",  #8
# "old_new/old_swimmer_highSubs", "old_new/gym_swimmer_highSubs", "old_new/gym_swimmer_highSubs_trainVar_S5",              #11
# "old_new/gym_swimmer_highSubs_highMaxiter_S5", "PILCO/gym_swimmer_highSubs_highMaxiter_longEval_S5",  #13
# "PILCO/gym_swimmer_best_replace"]                                                                           # 14
# num 12 and 13 have 10 runs. It's evaluated on double the number of steps (1000)


name = "evaluation_returns_full_safe_swimmer_final"
runs = 8;
for i in range(1, runs+1):
    #if name<13:
    rr = np.loadtxt( name + str(i) + ".csv", delimiter=',')

    if i == 1:
        N = rr.shape[0]
        r_means = np.zeros((runs, N))
        r_vars = np.zeros((runs, N))
    r_means[i-1, :] = np.mean(rr, 1)
    #print(r_means)
    r_vars[i-1, :] = np.std(rr, 1)
    #print(r_vars)

# Best so far

#Gym returns only for the ones that have them


# Best
bests = np.max(r_means, 1)
best_mean = np.mean(bests)
best_std = np.std(bests)
print(best_mean)
print(best_std)

means_per_iteration = np.mean(r_means, 0) # mean over random seeds
std_per_iteration = np.std(r_means, 0)

plt.plot(means_per_iteration, label='SafePILCO')
plt.fill_between(range(len(means_per_iteration)), means_per_iteration + std_per_iteration, means_per_iteration - std_per_iteration, alpha=0.3)
plt.savefig('current_iter_performance.png')
plt.show()


best_so_far = r_means.copy()
for seed in range(best_so_far.shape[0]):
    for iteration in range(1, best_so_far.shape[1]):
        if best_so_far[seed, iteration-1] < r_means[seed, iteration]:
            best_so_far[seed, iteration] = r_means[seed, iteration]
        else:
            best_so_far[seed, iteration] = best_so_far[seed, iteration-1]

means_best_so_far = np.mean(best_so_far, 0)
stds_best_so_far = np.std(best_so_far, 0)

plt.plot(means_best_so_far, label='SafePILCO')
plt.fill_between(range(len(means_best_so_far)), means_best_so_far + stds_best_so_far, means_best_so_far - stds_best_so_far, alpha=0.3)
plt.savefig('best_iter_performance.png')
plt.show()
