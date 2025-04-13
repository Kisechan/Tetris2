# tetris2_gym.py
import gym
from gym import spaces
import numpy as np
from tetris2_env import Tetris2_env

class Tetris2GymEnv(gym.Env):
    def __init__(self):
        super(Tetris2GymEnv, self).__init__()
        self.env = Tetris2_env()

        # 定义观察空间
        self.observation_space = spaces.Dict({
            "current_grid": spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.int32),
            "enemy_grid": spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.int32),
            "current_piece": spaces.Discrete(7),
            "enemy_piece": spaces.Discrete(7),
            "piece_counts": spaces.Box(low=0, high=100, shape=(7,), dtype=np.int32),
            "enemy_piece_counts": spaces.Box(low=0, high=100, shape=(7,), dtype=np.int32),
            "max_height": spaces.Discrete(21),
            "enemy_max_height": spaces.Discrete(21)
        })

        # 定义动作空间
        self.action_space = spaces.Tuple([
            spaces.Tuple([
                spaces.Discrete(10),  # x position (1-10)
                spaces.Discrete(20),  # y position (1-20)
                spaces.Discrete(4)    # rotation (0-3)
            ]),
            spaces.Discrete(7)       # next block (0-6)
        ])

    # 调试信息
    def reset(self):
        state = self.env.reset()
        print(f"Reset state type: {type(state)}")  # 应该是dict
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        print(f"Step state type: {type(next_state)}")  # 应该是dict
        return next_state, reward, done, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def get_valid_actions(self):
        return self.env.get_valid_actions()

    def close(self):
        pass

def register_env():
    from gym.envs.registration import register
    register(
        id='Tetris2-v0',
        entry_point='tetris2_gym:Tetris2GymEnv',
        max_episode_steps=200
    )
    print("注册完毕")