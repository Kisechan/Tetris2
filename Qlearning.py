from typing import ValuesView, List, Optional
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import numpy as np
import random
from typing import Optional, Iterable

from datetime import datetime
from statistics import mean
from tqdm import tqdm
from keras.engine.saving import save_model



class DQNAgent:
    def __init__(self,
                 state_size, # 输入状态维度数
                 mem_size = 10000,
                 discount_factor = 0.95, # 折扣因子
                 epsilon = 1.0, # 初始探索概率
                 epsilon_min = 0.0, # 最小探索概率
                 epsilon_rate = 500, # 探索率线性衰减
                 n_Linear = (32, 32), # 神经元数
                 activate = ('relu',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         'relu', 'linear'), # 激活函数
                 loss='mse', # 损失函数
                 optimizer='adam', # 优化器
                 replay_start_size=None # 回放内存达到多少时开始训练
                 ):
        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_rate
        self.n_Linear = n_Linear
        self.activations = activate
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model()

    def _build_model(self):
        # 构建神经网络
        model = Sequential()
        model.add(Dense(self.n_Linear[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_Linear)):
            model.add(Dense(self.n_Linear[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))
        model.compile(loss=self.loss, optimizer=self.optimizer)

    def add_to_memory(self, state, next_state, reward, done):
        # 当前状态 下一状态 奖励 和 是否结束
        self.memory.append((state, next_state, reward, done))

    def random_value(self):
        """Random score for a certain action"""
        return random.random()

    def predict_value(self, state: np.ndarray) -> float:
        """Predicts the score for a certain state"""
        return self.model.predict(state)[0]

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
            # 转化成一个 1 * state_size 的向量  其中 state_size 是特征维度数量

        if (random.random() < self.epsilon):
            return self.random_value()
        else:
            return self.predict_value(state)

    def best_state(self, states: ValuesView[List[int]]) -> List[int]:
        """Returns the best state for a given collection of states"""
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            max_value: Optional[float] = None
            best_state: Optional[List[int]] = None
            for state in states:
                # 询问神经网络最终的答案
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state
        return best_state

    def train(self, batch_size=32, epochs=3):
        n = len(self.memory)
        if n >= self.replay_start_size and n >= batch_size:
            batch = random.sample(self.memory, batch_size)
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

class AgentConf():
    def __init__(self):
        self.n_Linear = (32, 32)
        self.batch_size = 512
        self.activate = ('relu', 'relu', 'linear')
        self.episodes = 2000 # 总训练轮次
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_rate = 2000
        self.mem_size = 25000 # 经验回放大小
        self.discount = 0.99
        self.replay_start_size = 2000 # 在开始训练前需要的最小经验回放数
        self.epoch = 1 # 每轮训练时训练的epoch数
        self.train_every = 1  # 每多少轮训练一次
        self.log_every = 10  # 每多少轮记录一次日志
        self.max_steps: Optional[int] = 10000  # 每轮游戏的最大步数


