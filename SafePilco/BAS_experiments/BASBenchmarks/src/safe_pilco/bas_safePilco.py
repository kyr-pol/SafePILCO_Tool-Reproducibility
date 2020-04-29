import numpy as np
import matlab.engine
import sys
import os
from gpflow import set_trainable
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward, CombinedRewards
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import spaces
from pilco.safe_pilco.rewards_safe import SingleConstraint
from pilco.utils import rollout, policy
from pilco.safe_pilco.safe_pilco import SafePILCO


safe = int(sys.argv[1]) # turning this flag off (0) should the same experiment without safety considerations
seed = sys.argv[2]
np.random.seed(int(seed))
name = ""

# This can be used to run a different case study,not included in the paper (simpler environment)
class BASWrapper1():
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.CaseStudy1_Mda_safe_pilco(nargout=0)
        self.initial_state = [[15.0], [20.0], [35.0], [35.0]]
        self.x = np.array(self.initial_state.copy()) \
                          + 0.01 * np.random.normal(size=np.array(self.initial_state).shape)
        self._x_mat = matlab.double(self.initial_state)
        self.action_space  = spaces.Box(low=-10.0,high=10.0,shape=(1,))

    def step(self, action):
        self._x_mat = self.eng.my_step(self._x_mat, 20.0 + 10*np.float(action))
        print(self._x_mat)
        self.x = np.array(self._x_mat)
        return np.reshape(self.x.copy(), (4,)), 0.0, False, {}

    def reset(self):
        self.eng.CaseStudy1_Mda_safe_pilco(nargout=0)
        self.x = np.array(self.initial_state.copy()) \
                 + 0.01 * np.random.normal(size=np.array(self.initial_state).shape)
        self._x_mat = matlab.double(self.initial_state)
        return np.reshape(self.x.copy(), (4,))

    def render(self):
        pass

    def close(self):
        # VERY IMPORTANT
        self.eng.quit()

class BASWrapper2():
    def __init__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.CaseStudy2_Mc_safe_pilco(nargout=0)
        self.initial_state = [ [15], [20], [18], [18], [18], [18], [18]]
        self.x = np.array(self.initial_state.copy()) \
                          + 0.01 * np.random.normal(size=np.array(self.initial_state).shape)
        self._x_mat = matlab.double(self.initial_state)
        self.action_space  = spaces.Box(low=-10.0,high=10.0,shape=(1,))

    def step(self, action):
        self._x_mat = self.eng.my_step(self._x_mat, 20+ 10*np.float(action))
        self.x = np.array(self._x_mat)
        r = -((self.x[0] - 20.0)**2 + (self.x[1] - 20.0)**2)/10
        return np.reshape(self.x.copy(), (7,)), r, False, {}

    def reset(self):
        self.eng.CaseStudy2_Mc_safe_pilco(nargout=0)
        self.x = np.array(self.initial_state.copy()) \
                          + 0.01 * np.random.normal(size=np.array(self.initial_state).shape)
        self._x_mat = matlab.double(self.initial_state)
        return np.reshape(self.x.copy(), (7,))

    def render(self):
        pass

    def close(self):
        # VERY IMPORTANT
        self.eng.quit()

J = 4
T = 12 * 4
experim_num = 2
if experim_num == 1:
    # for a step
    m_init = np.reshape([15.0, 20.0, 35.0, 35.0], (1,4))
    S_init = 0.2 * np.eye(4)
    targets = np.array([20.0, 20.0, 35.0, 35.0])
    weights = 1e-3 * np.eye(4)
    weights[0,0] = 1.0; weights[1,1] = 1.0
    env = BASWrapper1()
    max_temp = 20.5
else:
    m_init = np.reshape([15.0, 20.0, 18.0, 18.0, 18.0, 18.0, 18.0], (1,7))
    S_init = 0.2 * np.eye(7)
    targets = np.array([20.0, 20.0, 18.0, 18.0, 18.0, 18.0, 18.0])
    weights = 1e-3 * np.eye(7)
    weights[0,0] = 1.0; weights[1,1] = 1.0
    env = BASWrapper2()
    max_temp = 20.5

X,Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=T)
for i in range(1,J):
    X_, Y_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=T)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim

controller = LinearController(state_dim=state_dim, control_dim=control_dim,
                              max_action=1.0, t=targets)

if safe:
    if experim_num==1:
        C1 = SingleConstraint(0, high=max_temp, inside=False)
    else:
        C1 = SingleConstraint(1, high=max_temp, inside=False)
    R1 = ExponentialReward(state_dim=state_dim, t=targets)
    R = CombinedRewards(state_dim, [R1, C1], coefs=[1.0, -15.0])
else:
    R1 = ExponentialReward(state_dim=state_dim, t=targets, W=weights)
    t2 = targets.copy(); t2[1] = 20.5
    w2 = weights.copy(); w2[1,1] = 4.0
    R2 = ExponentialReward(state_dim=state_dim, t=t2, W=w2)
    R = CombinedRewards(state_dim, [R1, R2], coefs=[1.0, -2.0])

pilco = SafePILCO((X, Y), controller=controller, horizon=T, reward_add=R1, reward_mult=C1,
                  m_init=m_init, S_init=S_init, mu=-50.0)
for model in pilco.mgpr.models:
    model.likelihood.variance.assign(0.1)
    set_trainable(model.likelihood.variance, False)

N=3
th = 0.05
eval_runs = 5
evaluation_returns_full = np.zeros((N, eval_runs))
evaluation_returns_sampled = np.zeros((N, eval_runs))
X_eval = []
for rollouts in range(N):
    print("*********ITERATION***********")
    pilco.optimize_models()
    pilco.optimize_policy(maxiter=50)
    if safe:
        m_p = np.zeros((T, state_dim))
        S_p = np.zeros((T, state_dim, state_dim))
        predicted_risks = np.zeros(T)
        m_ = m_init.copy()
        S_ = S_init.copy()
        for h in range(T):
            m_h, S_h, _ = pilco.predict(m_init, S_init, h)
            m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
            predicted_risks[h], _ = C1.compute_reward(m_h, S_h)
            #predicted_rewards[h], _ = R1.compute_reward(m_h, S_h)
        estimate_risk = 1 - np.prod(1.0-predicted_risks)
        print(estimate_risk)
        if estimate_risk < th:
            if estimate_risk < th/4:
                pilco.mu.assign(0.75 * pilco.mu.value())
            X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, timesteps=T)
            X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
            pilco.mgpr.set_data((X, Y))
            for k in range(0, eval_runs):
                [X_eval_, _,
                evaluation_returns_sampled[rollouts, k],
                evaluation_returns_full[rollouts, k]] = rollout(env, pilco,
                                                               timesteps=T,
                                                               verbose=False, SUBS=1,
                                                               render=False)
                if len(X_eval)==0:
                    X_eval = X_eval_.copy()
                else:
                    X_eval = np.vstack((X_eval, X_eval_))
            np.savetxt("../../../results/X_" + seed + ".csv", X, delimiter=',')
            np.savetxt("../../../results/X_eval_" + seed + ".csv", X_eval, delimiter=',')
            np.savetxt("../../../results/evaluation_returns_sampled_"  + seed + ".csv", evaluation_returns_sampled, delimiter=',')
            np.savetxt("../../../results/evaluation_returns_full_" + seed + ".csv", evaluation_returns_full, delimiter=',')
        else:
            print("*********CHANGING***********")
            X_2, Y_2, _, _ = rollout(env, pilco, timesteps=T, verbose=True)
            _, _, r = pilco.predict(m_init, S_init, T)
            print("Before ", r)
            if estimate_risk > th:
                pilco.mu.assign(1.5 * pilco.mu.value())
            _, _, r = pilco.predict(m_init, S_init, T)
            print("After ", r)
    else:
        X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, timesteps=T)
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
