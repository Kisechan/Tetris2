import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tetris2_env import Tetris2Env

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
    STATE_DIM = 20*10 + 7 + 7 + 1 + 1  # 网格(200) + 当前方块(7) + 对方统计(7) + 连消(1) + 高度(1)

# PPO策略网络
class TetrisPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(Config.STATE_DIM, Config.HIDDEN_SIZE),
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
        features = self.shared(x)

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
        self.env = Tetris2Env()

    def get_legal_actions(self, color):
        """获取当前所有合法动作"""
        legal_placements = []
        block_type = self.env.current_type[color]

        # 检查所有可能的放置位置
        for x in range(1, 11):
            for o in range(4):
                # 找到能落下的最低y位置
                for y in range(self.env.MAPHEIGHT, 0, -1):
                    block = self.env.Tetris(self.env, block_type, color)
                    if (block.set_pos(x, y, o).is_valid() and
                            self.env.check_direct_drop(color, block_type, x, y, o)):
                        legal_placements.append((x, o))
                        break

        # 检查合法的方块类型选择
        enemy_counts = self.env.type_count[1 - color]
        min_count = min(enemy_counts)
        legal_blocks = [t for t in range(7) if enemy_counts[t] <= min_count + 2]

        return legal_placements, legal_blocks

    def select_action(self, state, color):
        """根据策略选择动作，应用动作屏蔽"""
        legal_placements, legal_blocks = self.get_legal_actions(color)

        with torch.no_grad():
            output = self.policy(torch.FloatTensor(state).unsqueeze(0))  # 确保有batch维度

        # 应用动作屏蔽
        # 1. 处理放置动作
        place_probs = np.zeros((10, 4))
        for x, o in legal_placements:
            place_probs[x-1, o] = output['placement'][0, x-1, o].item()
        if place_probs.sum() > 0:
            place_probs = place_probs / place_probs.sum()  # 更安全的归一化方式

        # 2. 处理方块选择 - 修复索引
        block_probs = np.zeros(7)
        for t in legal_blocks:
            block_probs[t] = output['block_select'][0][t].item()  # 正确的索引方式
        if block_probs.sum() > 0:
            block_probs = block_probs / block_probs.sum()  # 更安全的归一化方式

        # 采样动作
        if len(legal_placements) == 0 or len(legal_blocks) == 0:
            return None, 0, 0

        # 处理可能的数值问题
        place_probs = np.nan_to_num(place_probs, nan=0.0, posinf=0.0, neginf=0.0)
        block_probs = np.nan_to_num(block_probs, nan=0.0, posinf=0.0, neginf=0.0)

        if place_probs.sum() <= 0:
            place_probs = np.ones_like(place_probs) / place_probs.size
        if block_probs.sum() <= 0:
            block_probs = np.ones_like(block_probs) / block_probs.size

        # 采样放置动作
        place_flat = place_probs.flatten()
        place_idx = np.random.choice(len(place_flat), p=place_flat)
        x = place_idx // 4 + 1
        o = place_idx % 4

        # 找到对应的y坐标
        y = self.env.MAPHEIGHT
        block = self.env.Tetris(self.env, self.env.current_type[color], color)
        while y >= 1:
            if block.set_pos(x, y, o).is_valid():
                break
            y -= 1

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

        # 从缓冲区采样
        samples = self.buffer.sample(Config.BATCH_SIZE)
        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*samples)

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        old_log_probs = torch.FloatTensor(np.array(old_log_probs))
        values = torch.FloatTensor(np.array(values))

        # 计算GAE和回报
        returns = self.compute_returns(rewards, dones, values)
        advantages = returns - values

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(Config.PPO_EPOCHS):
            for idx in range(0, Config.BATCH_SIZE, Config.MINIBATCH_SIZE):
                # 获取小批量
                mb_states = states[idx:idx+Config.MINIBATCH_SIZE]

                # 一次性计算所有输出的概率
                output = self.policy(mb_states)

                # 计算新概率 - 修改这部分
                mb_actions = actions[idx:idx+Config.MINIBATCH_SIZE]
                new_log_probs = []
                for i, action in enumerate(mb_actions):
                    # 为每个动作创建对应的输出切片
                    single_output = {
                        'placement': output['placement'][i:i+1],
                        'block_select': output['block_select'][i:i+1],
                        'value': output['value'][i:i+1]
                    }
                    new_log_probs.append(self.compute_log_prob(single_output, action))
                new_log_probs = torch.stack(new_log_probs)

                # 其余部分保持不变
                mb_old_log_probs = old_log_probs[idx:idx+Config.MINIBATCH_SIZE]
                mb_advantages = advantages[idx:idx+Config.MINIBATCH_SIZE]
                mb_returns = returns[idx:idx+Config.MINIBATCH_SIZE]

                # 计算比率
                ratios = torch.exp(new_log_probs - mb_old_log_probs)

                # 计算损失
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1-Config.CLIP_EPSILON, 1+Config.CLIP_EPSILON) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = (mb_returns - output['value'].squeeze()).pow(2).mean()

                # 熵奖励
                entropy = -(output['placement'] * torch.log(output['placement'] + 1e-10)).mean() + \
                          -(output['block_select'] * torch.log(output['block_select'] + 1e-10)).mean()

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
            self.env.reset()
            state = self.env.get_state(0)  # 玩家0
            total_reward = 0
            done = False

            while not done:
                # 选择动作
                action, value, log_prob = self.select_action(state, 0)
                if action is None:  # 没有合法动作
                    print("没有合法动作")
                    reward = -10
                    done = True
                else:
                    # 执行动作
                    reward, done = self.env.step(0, (action['block'], action['x'], action['y'], action['o']))
                    next_state = self.env.get_state(0)

                # 存储经验
                self.buffer.add(state, action, reward, next_state if not done else None, done, log_prob, value)

                # 更新状态
                state = next_state
                total_reward += reward

                # 更新策略
                if len(self.buffer) >= Config.BATCH_SIZE:
                    self.update()

            # 保存模型
            if episode % Config.SAVE_INTERVAL == 0 and episode != 0:
                torch.save(self.policy.state_dict(), f'tetris_ppo_{episode}.pth')

            print(f"Episode {episode + 1}, Reward: {total_reward}")

if __name__ == "__main__":
    print("训练开始")
    agent = PPOTetrisAgent()
    agent.train()