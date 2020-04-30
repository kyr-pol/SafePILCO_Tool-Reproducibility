import numpy as np
import gym


class DoublePendWrapper():
    def __init__(self):
        self.env = gym.make('InvertedDoublePendulum-v2').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def state_trans(self, s):
        a1 = np.arctan2(s[1], s[3])
        a2 = np.arctan2(s[2], s[4])
        s_new = np.hstack([s[0], a1, a2, s[5:-3]])
        return s_new

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        if np.abs(ob[0])> 0.90 or np.abs(ob[-3]) > 0.15 or  np.abs(ob[-2]) > 0.15 or np.abs(ob[-1]) > 0.15:
            done = True
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()


class myPendulum():
    def __init__(self):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(low=0, high=0.01*high) # only difference
        self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()
