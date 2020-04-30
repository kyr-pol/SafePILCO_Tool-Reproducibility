import numpy as np
import sys
from matplotlib import pyplot as plt
import seaborn as sns

#from calc_transformed_bounds_cars import calc_bounds

def check_for_swimmer(X, bounds):
    violation = False
    for i in range(len(X)):
        if X[i,3] < -bounds or X[i,3] > bounds or X[i,4] < -bounds or X[i,4] > bounds:
            violation = True
    return violation

#sns.set_context('paper')
sns.set(font_scale=1.5)
#plt.rcParams.update({'font.size': 20})
sns.set_style('white')


runs_so_far = 10
T = 25
J = 10
# for which_env in range(1):
X = []
X_eval =[]
all_returns_sampled = []
all_returns_full = []
path = 'res/'

name = ''
for i in range(0, runs_so_far):
    X.append(np.loadtxt('results/X_' + name  + str(i) + '.csv', delimiter=','))
    X_eval.append(np.loadtxt('results/X_eval_' + name  + str(i) + '.csv', delimiter=','))
    all_returns_sampled.append(np.loadtxt('results/evaluation_returns_sampled_'+ name  + str(i) + '.csv', delimiter=','))
    all_returns_full.append(np.loadtxt('results/evaluation_returns_full_' + name  + str(i) + '.csv', delimiter=','))

#all_bounds = calc_bounds()
bound = (100 / 180 * np.pi) * 0.95

best_no_collision_returns = []
all_runs_best_means = []
all_runs_best_stds = []
all_runs_collisions = []
all_interactions_number = []
for run in range(runs_so_far):
    run_bounds = bound
    #X1 = X[i][T*J:,:] # initial random actions we don't care about

    interactions = (len(X[run]) - T*J) // T
    starting_indices = [inter*T + T*J for inter in range(interactions)]

    collision_array = []
    collisions = 0
    for j in starting_indices:
        collision_array.append(check_for_swimmer(X[run][j:j+T, :], run_bounds))
        if collision_array[-1]:
            collisions += 1

    returns_filtered = all_returns_full[run][~np.all( all_returns_full[run]== 0, axis=1)]
    episodes_without_collisions = []
    eval_runs = 0
    best_no_collision_return = -np.inf
    for k in range(interactions):
        if not collision_array[k]:
            eval_runs += 1
            if np.mean(returns_filtered[k,:]) > best_no_collision_return:
                best_no_collision_returns = np.mean(returns_filtered[k,:])

    all_runs_best_means.append(best_no_collision_returns)
    all_runs_collisions.append(collisions)
    all_interactions_number.append(interactions)

print("Mean best return: ", np.mean(all_runs_best_means))
print("Std best return: ", np.std(all_runs_best_means))
print("Mean number of collisions: ", np.mean(all_runs_collisions))
print("Std number of collisions: ", np.std(all_runs_collisions))
print("Mean number of interactions blocked: ", all_returns_full[0].shape[0] - np.mean(all_interactions_number))
print("Std number of interactions blocked: ", np.std(all_interactions_number))
