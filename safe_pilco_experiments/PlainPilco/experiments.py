from master_alg import pilco_run
from safe_pilco_experiments.utils import policy, rollout, Normalised_Env, DoublePendWrapper, myPendulum
import numpy as np
import gym

number_of_random_seeds = 10

# Inverted Pendulum
env = gym.make('InvertedPendulum-v2')
for i in range(0, number_of_random_seeds):
    name = "inverted_pendulum/"
    pilco_run(env, 3, 5, logging=True, eval_runs=10,
              eval_max_timesteps=100, seed=i,
              fixed_noise=0.001, name=name)

# Mountain Car
SUBS=5
T = 25
for i in range(0, number_of_random_seeds):
    env = gym.make('MountainCarContinuous-v0')
    name = 'mountain_car/'
    # Normalise before calling pilco_run
    # Initial random rollouts to generate a dataset
    X1, Y1, _, _ = rollout(env=env, pilco=None, random=True, timesteps=T, SUBS=SUBS)
    for j in range(1,5):
        X1_, Y1_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=T, SUBS=SUBS)
        X1 = np.vstack((X1, X1_))
        Y1 = np.vstack((Y1, Y1_))
    #env.close()
    env = Normalised_Env('MountainCarContinuous-v0', np.mean(X1[:,:2],0), np.std(X1[:,:2], 0))
    controller = {'type':'rbf', 'basis_functions':25}
    reward = {'type':'exp',
              't':np.divide([0.5,0.0] - env.m, env.std),
              'W':np.diag([0.5, 0.1])}
    pilco_run(env, 4, 2,
              SUBS=SUBS,
              restarts=3,
              maxiter=50,
              cont=controller,
              rew=reward,
              sim_timesteps=T,
              plan_timesteps=T,
              fixed_noise=0.05,
              m_init=np.reshape(np.divide(X1[0,:-1]-env.m, env.std), (1,2)),
              S_init=0.5 * np.eye(2),
              logging=True,
              eval_runs=5,
              eval_max_timesteps=T,
              name=name,
              seed=i)


# Pendulum Swing Up
env = myPendulum()
for i in range(0, number_of_random_seeds):
    name = 'pend_swing_up/'
    controller = {'type':'rbf', 'basis_functions':30, 'max_action':2.0}
    reward = {'type':'exp', 't':np.array([1.0, 0.0, 0.0]), 'W':np.diag([2.0, 2.0, 0.3])}
    T = 40
    name = 'pend_swing_up/'
    pilco_run(env, 8, 4,
              SUBS=3,
              maxiter=50,
              restarts=2,
              m_init = np.reshape([-1.0, 0, 0.0], (1,3)),
              S_init = np.diag([0.01, 0.05, 0.01]),
              cont=controller,
              rew=reward,
              sim_timesteps=T,
              plan_timesteps=T,
              logging=True,
              eval_runs=5,
              eval_max_timesteps=T,
              fixed_noise=0.001,
              name=name,
              seed=i)

# Inverted Double Pendulum
env = DoublePendWrapper()
for i in range(0, number_of_random_seeds):
    controller = {'type':'rbf', 'basis_functions':40, 'max_action':1.0}
    state_dim = 6
    weights = 5.0 * np.eye(state_dim)
    weights[0,0] = 1.0
    weights[3,3] = 1.0
    reward = {'type':'exp', 't':np.zeros(state_dim), 'W':weights}
    name = 'double_pendulum/'
    pilco_run(env, 10, 5,
              SUBS = 1,
              maxiter=100,
              m_init = np.zeros(state_dim)[None, :],
              S_init = 0.005 * np.eye(state_dim),
              cont=controller,
              rew=reward,
              sim_timesteps = 130,
              plan_timesteps = 40,
              restarts=2,
              logging=True,
              eval_runs=5,
              eval_max_timesteps=130,
              fixed_noise=0.001,
              name=name,
              seed=i)
