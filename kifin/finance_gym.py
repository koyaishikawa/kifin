import gym
from gym.spaces import Discrete, Box
import numpy as np


class FinanceGym(gym.Env):
    def __init__(self, input_data, output_data, cost=0.0):
        self.input_data = input_data
        self.output_data = output_data
        self.data_length = input_data.shape[0]
        self.cost = cost

        _obs = self.reset()
        # TODO observation_space, action_space
        self.action_space = Discrete(2)
        self.observation_space = Box(input_data.min(), input_data.max(), _obs.shape)

    def __len__(self):
        return self.input_data.shape[0]
        
    def step(self, action):
        """
        action
        0 : sell
        1 : hold
        2 : buy
        """
        self.reward = self.get_reward(action)
        self.done = self.get_done()

        self._t += 1
        self.obs = np.append(self.input_data[self._t].flatten(), self.prev_action)
        self.info = ""
        self.prev_action = action
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self._t = 0
        self.prev_action = 1
        self.obs = np.append(self.input_data[self._t].flatten(), self.prev_action)
        self.total_diff = 0
        return self.obs

    def get_reward(self, action):
        # 取引継続 or 取引なし
        if self.prev_action - action == 0:
            if action == 1:
                reward = 0.0
            else:
                self.total_diff += self.output_data[self._t].item()
                reward = 0.0
                
        # 取引開始 or 取引終了    
        else: 
            if action == 1:
                reward = self.total_diff * (self.prev_action - 1)
                self.total_diff = 0
                
            else:
                reward = abs(action - self.prev_action) * self.cost * (-1)

                if self.total_diff == 0:
                    self.total_diff += self.output_data[self._t].item()
                else:
                    reward += self.total_diff * (self.prev_action - 1)
                    self.total_diff = self.output_data[self._t].item()
        
        return reward   

    def get_done(self):
        return not (self.data_length - self._t > 2)
