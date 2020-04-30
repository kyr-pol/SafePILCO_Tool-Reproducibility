import numpy as np
import gym
from pilco.utils import policy, rollout
from linear_cars_env import LinearCars


def calc_bounds():
    env = LinearCars()
    runs = 10
    J = 5
    T = 25

    means = np.zeros((runs, 4))
    stds = np.zeros((runs, 4))
    for i in range(runs):
        np.random.seed(i)

        for j in range(J):
            if j==0:
                X, _, _, _ = rollout(env, None, timesteps=T, random=True, verbose=False)
            else:
                X_, _, _, _ = rollout(env, None, timesteps=T, random=True, verbose=False)
                X = np.vstack((X, X_))

        means_run = np.mean(X[:, :-1], 0)
        stds_run = np.std(X[:, :-1], 0)
        means[i,:] = means_run[:]
        stds[i, :] = stds_run[:]

    bounds = np.zeros((runs, 2, 2)) # runs, car, lower/upper
    for i in range(runs):
        bound_x1 = 1 / stds[i, 0]
        bound_x2 = 1 / stds[i, 2]
        bounds[i, 0, 0] = -bound_x1 - means[i, 0]/stds[i, 0]
        bounds[i, 0, 1] = bound_x1 - means[i, 0]/stds[i, 0]
        bounds[i, 1, 0] = -bound_x2 - means[i, 1]/stds[i, 1]
        bounds[i, 1, 1] = bound_x2 - means[i, 1]/stds[i, 1]
    return bounds


#
# X = np.zeros((J*Ts[0], 5))
# means = np.mean(X, 1) # average over the 5 rollouts
# env_m1 = X[:,0] #car 1
# env_m2 = X
if __name__=='__main__':
    calc_bounds()
