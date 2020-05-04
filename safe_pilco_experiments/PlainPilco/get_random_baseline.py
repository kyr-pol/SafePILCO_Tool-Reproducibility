import numpy as np
import gym
from safe_pilco_experiments.utils import policy, rollout, myPendulum, DoublePendWrapper

envs = [gym.make('MountainCarContinuous-v0'), gym.make('InvertedPendulum-v2'), myPendulum(), DoublePendWrapper(),
        gym.make('Swimmer-v2')]
Ts = [25, 40, 40, 130, 200]
SUBSs = [5, 1, 3, 1, 5]
J = 50
all_means = []
all_stds = []

for i in range(len(envs)):
    rets_full = []
    env = envs[i]
    for k in range(0, J):
        _, _, ret_sam, ret_full = rollout(env, None, timesteps=Ts[i], random=True, SUBS=SUBSs[i], verbose=False)
        rets_full.append(ret_full)
    all_means.append(np.mean(rets_full))
    all_stds.append(np.std(rets_full))
    #env.close()

np.savetxt("random_policy.csv", np.vstack((all_means, all_stds)), delimiter=',')
