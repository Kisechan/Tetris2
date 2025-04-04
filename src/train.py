import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tetris2 import Tetris, MAPWIDTH, MAPHEIGHT

# 超参数配置
class Config:
    GAMMA = 0.99
    LAMBDA = 0.95
    LR = 3e-4
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    BATCH_SIZE = 64
    MINIBATCH_SIZE = 16
    PPO_EPOCHS = 4
    MAX_EPISODES = 10000
    SAVE_INTERVAL = 100
    HIDDEN_SIZE = 256

# 神经网络模型
class TetrisPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 状态编码层
        self.encoder = nn.Sequential(
            nn.Linear(20*10 + 7 + 7 + 1 + 1, Config.HIDDEN_SIZE),  # 网格(200) + 当前方块(7) + 对方统计(7) + 连消(1) + 高度(1)
            nn.ReLU()
        )

        # 放置动作头 (x位置和旋转)
        self.placement_head = nn.Sequential(
            nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_SIZE, 10*4)  # 10列 × 4种旋转
        )

        # 方块选择头
        self.block_head = nn.Sequential(
            nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_SIZE, 7)  # 7种方块类型
        )

        # 价值函数头
        self.value_head = nn.Linear(Config.HIDDEN_SIZE, 1)

    def forward(self, x):
        features = self.encoder(x)

        # 获取各动作的概率分布
        place_logits = self.placement_head(features).view(-1, 10, 4)
        block_logits = self.block_head(features)

        return {
            'placement': torch.softmax(place_logits, dim=-1),
            'block_select': torch.softmax(block_logits, dim=-1),
            'value': self.value_head(features)
        }

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, old_log_prob, value):
        self.buffer.append((state, action, reward, next_state, done, old_log_prob, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

# PPO智能体
class PPOTetrisAgent:
    def __init__(self):
        self.policy = TetrisPolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=Config.LR)
        self.buffer = ReplayBuffer(10000)
        self.game = Tetris()

    def get_state(self, color):
        """将游戏状态编码为神经网络输入"""
        # 网格状态 (20行 × 10列)
        grid = self.game.gridInfo[color][1:21, 1:11].flatten()

        # 当前方块类型 (one-hot)
        current_block = np.zeros(7)
        current_block[self.game.nextTypeForColor[color]] = 1

        # 对方方块统计
        enemy_stats = np.array(self.game.typeCountForColor[1 - color])

        # 连消计数和最大高度
        combo = np.array([self.game.elimCombo[color]])
        height = np.array([self.game.maxHeight[color]])

        return np.concatenate([grid, current_block, enemy_stats, combo, height])

    def get_legal_actions(self, color):
        """获取当前所有合法动作"""
        legal_placements = []
        block_type = self.game.nextTypeForColor[color]

        # 检查所有可能的放置位置
        for x in range(1, 11):
            for o in range(4):
                # 找到能落下的最低y位置
                for y in range(MAPHEIGHT, 0, -1):
                    if self.game.checkValidPlacement(color, x, y, o, block_type):
                        legal_placements.append((x, o))
                        break

        # 检查合法的方块类型选择
        enemy_counts = self.game.typeCountForColor[1 - color]
        min_count = min(enemy_counts)
        legal_blocks = [t for t in range(7) if enemy_counts[t] <= min_count + 2]

        return legal_placements, legal_blocks

    def select_action(self, state, color):
        """根据策略选择动作，应用动作屏蔽"""
        legal_placements, legal_blocks = self.get_legal_actions(color)

        with torch.no_grad():
            output = self.policy(torch.FloatTensor(state))

        # 应用动作屏蔽
        # 1. 处理放置动作
        place_probs = np.zeros((10, 4))
        for x, o in legal_placements:
            place_probs[x-1, o] = output['placement'][0, x-1, o].item()
        place_probs /= place_probs.sum()

        # 2. 处理方块选择
        block_probs = np.zeros(7)
        for t in legal_blocks:
            block_probs[t] = output['block_select'][0, t].item()
        block_probs /= block_probs.sum()

        # 采样动作
        place_flat = place_probs.flatten()
        place_idx = np.random.choice(len(place_flat), p=place_flat)
        x = place_idx // 4 + 1
        o = place_idx % 4

        # 找到对应的y坐标
        y = self.game.findLowestY(color, x, o, self.game.nextTypeForColor[color])

        # 选择方块类型
        block_type = np.random.choice(7, p=block_probs)

        # 计算动作的对数概率
        action = {'x': x, 'y': y, 'o': o, 'block': block_type}
        log_prob = self.compute_log_prob(output, action)

        return action, output['value'].item(), log_prob

    def compute_log_prob(self, output, action):
        """计算动作的对数概率"""
        # 放置部分
        place_log_prob = torch.log(output['placement'][0, action['x']-1, action['o']])

        # 方块选择部分
        block_log_prob = torch.log(output['block_select'][0, action['block']])

        return place_log_prob + block_log_prob

    def update(self):
        """PPO算法更新"""
        if len(self.buffer) < Config.BATCH_SIZE:
            return

        # 计算GAE和回报
        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*self.buffer.sample(Config.BATCH_SIZE))

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = actions  # 保持字典形式
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        values = torch.FloatTensor(np.array(values))

        # 计算优势
        returns = self.compute_returns(rewards, dones, values)
        advantages = returns - values

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(Config.PPO_EPOCHS):
            for idx in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
                # 获取小批量
                mb_states = states[idx:idx+Config.MINIBATCH_SIZE]
                mb_actions = actions[idx:idx+Config.MINIBATCH_SIZE]
                mb_old_log_probs = old_log_probs[idx:idx+Config.MINIBATCH_SIZE]
                mb_advantages = advantages[idx:idx+Config.MINIBATCH_SIZE]
                mb_returns = returns[idx:idx+Config.MINIBATCH_SIZE]

                # 计算新概率
                output = self.policy(mb_states)
                new_log_probs = torch.stack([self.compute_log_prob(output[i:i+1], a)
                                             for i, a in enumerate(mb_actions)])

                # 计算比率
                ratios = torch.exp(new_log_probs - mb_old_log_probs)

                # 计算损失
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1-Config.CLIP_EPSILON, 1+Config.CLIP_EPSILON) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = (mb_returns - output['value'].squeeze()).pow(2).mean()

                # 熵奖励
                entropy = -(output['placement'] * torch.log(output['placement'])).mean() + \
                          -(output['block_select'] * torch.log(output['block_select'])).mean()

                # 总损失
                loss = policy_loss + 0.5 * value_loss - Config.ENTROPY_COEF * entropy

                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_returns(self, rewards, dones, values):
        """计算GAE回报"""
        returns = []
        R = 0
        for r, done, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = r + Config.GAMMA * R * (1 - done)
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def train(self):
        """训练循环"""
        for episode in range(Config.MAX_EPISODES):
            state = self.get_state(0)  # 假设当前玩家是0
            total_reward = 0
            done = False

            while not done:
                # 选择动作
                action, value, log_prob = self.select_action(state, 0)

                # 执行动作
                reward, done = self.game.step(0, action['x'], action['y'], action['o'], action['block'])
                next_state = self.get_state(0)

                # 存储经验
                self.buffer.add(state, action, reward, next_state, done, log_prob.item(), value)

                # 更新状态
                state = next_state
                total_reward += reward

                # 更新策略
                if len(self.buffer) >= Config.BATCH_SIZE:
                    self.update()

            # 保存模型
            if episode % Config.SAVE_INTERVAL == 0:
                torch.save(self.policy.state_dict(), f'tetris_ppo_{episode}.pth')
                print(f"Episode {episode}, Reward: {total_reward}")

if __name__ == "__main__":
    agent = PPOTetrisAgent()
    agent.train()