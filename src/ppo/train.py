import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from tetris2_env import Tetris2_env

class PPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__()

        # 棋盘编码器 (处理两个棋盘)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 计算卷积后的尺寸
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size + 2*padding - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2])))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1])))
        linear_input_size = convw * convh * 64

        # 方块类型和统计信息编码
        self.piece_embedding = nn.Embedding(7, 16)  # 7种方块类型
        self.stat_fc = nn.Linear(14 + 2, 64)  # 14个计数+2个高度

        # 合并所有特征
        self.combine_fc = nn.Linear(linear_input_size * 2 + 16 + 64, 512)

        # 输出头
        self.placement_head = nn.Linear(512, input_shape[1] * input_shape[2] * 4)  # 每个位置和旋转的概率
        self.next_block_head = nn.Linear(512, 7)  # 选择下一个方块
        self.value_head = nn.Linear(512, 1)  # 状态价值

    def forward(self, state):
        # 处理当前棋盘
        current_grid = state["current_grid"].unsqueeze(1).float()  # [batch, 1, H, W]
        c1 = torch.relu(self.conv1(current_grid))
        c2 = torch.relu(self.conv2(c1))
        c3 = torch.relu(self.conv3(c2))
        current_features = c3.view(c3.size(0), -1)

        # 处理对手棋盘
        enemy_grid = state["enemy_grid"].unsqueeze(1).float()
        e1 = torch.relu(self.conv1(enemy_grid))
        e2 = torch.relu(self.conv2(e1))
        e3 = torch.relu(self.conv3(e2))
        enemy_features = e3.view(e3.size(0), -1)

        # 处理当前方块
        piece_emb = self.piece_embedding(state["current_piece"].long())

        # 处理统计信息
        stats = torch.cat([
            state["piece_counts"].float(),
            state["enemy_piece_counts"].float(),
            state["max_height"].unsqueeze(1).float(),
            state["enemy_max_height"].unsqueeze(1).float()
        ], dim=1)
        stats_features = torch.relu(self.stat_fc(stats))

        # 合并所有特征
        combined = torch.cat([current_features, enemy_features, piece_emb, stats_features], dim=1)
        combined = torch.relu(self.combine_fc(combined))

        # 计算输出
        placement_logits = self.placement_head(combined)
        next_block_logits = self.next_block_head(combined)
        value = self.value_head(combined)

        return placement_logits, next_block_logits, value

class PPOAgent:
    def __init__(self, input_shape, num_actions, lr=3e-4, gamma=0.99, clip_epsilon=0.2, beta=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.memory = deque(maxlen=10000)

    def get_action(self, state):
        state = self._preprocess_state(state)
        with torch.no_grad():
            placement_logits, next_block_logits, value = self.policy(state)

            # 处理放置动作
            placement_probs = torch.softmax(placement_logits.view(-1), dim=0)
            placement_dist = Categorical(placement_probs)
            placement_action_flat = placement_dist.sample()

            # 将扁平化动作转换为(x,y,rotation)
            width, height = state["current_grid"].shape[-2:]
            rotation_size = 4
            placement_action = (
                placement_action_flat % width + 1,
                (placement_action_flat // width) % height + 1,
                (placement_action_flat // (width * height)) % rotation_size
            )

            # 处理下一个方块选择
            next_block_probs = torch.softmax(next_block_logits, dim=-1)
            next_block_dist = Categorical(next_block_probs)
            next_block_action = next_block_dist.sample()

            # 计算动作概率
            action_prob = placement_dist.log_prob(placement_action_flat).exp() * \
                          next_block_dist.log_prob(next_block_action).exp()

            return (placement_action, next_block_action.item()), action_prob.item(), value.item()

    def store_transition(self, state, action, prob, value, reward, done):
        self.memory.append((state, action, prob, value, reward, done))

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # 从记忆中采样
        samples = random.sample(self.memory, batch_size)
        states, actions, old_probs, old_values, rewards, dones = zip(*samples)

        # 预处理状态
        states = [self._preprocess_state(s) for s in states]

        # 计算折扣回报和优势
        returns = []
        advantages = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
            advantages.insert(0, R - old_values[i])

        # 标准化优势
        advantages = torch.tensor(advantages, device=self.device).float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换数据为张量
        returns = torch.tensor(returns, device=self.device).float()
        old_probs = torch.tensor(old_probs, device=self.device).float()
        old_values = torch.tensor(old_values, device=self.device).float()

        # 计算新概率和值
        placement_actions = []
        next_block_actions = []
        for action in actions:
            placement, next_block = action
            width = states[0]["current_grid"].shape[-1]
            height = states[0]["current_grid"].shape[-2]
            placement_flat = (placement[1]-1)*width*4 + (placement[0]-1)*4 + placement[2]
            placement_actions.append(placement_flat)
            next_block_actions.append(next_block)

        placement_actions = torch.tensor(placement_actions, device=self.device).long()
        next_block_actions = torch.tensor(next_block_actions, device=self.device).long()

        placement_logits, next_block_logits, values = self.policy(states)

        # 计算新概率
        placement_probs = torch.softmax(placement_logits.view(batch_size, -1), dim=-1)
        placement_dist = Categorical(placement_probs)
        placement_log_probs = placement_dist.log_prob(placement_actions)

        next_block_probs = torch.softmax(next_block_logits, dim=-1)
        next_block_dist = Categorical(next_block_probs)
        next_block_log_probs = next_block_dist.log_prob(next_block_actions)

        new_probs = (placement_log_probs + next_block_log_probs).exp()

        # PPO比率
        ratios = new_probs / old_probs

        # PPO目标函数
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(values.squeeze(), returns)

        # 熵奖励
        placement_entropy = placement_dist.entropy().mean()
        next_block_entropy = next_block_dist.entropy().mean()
        entropy = placement_entropy + next_block_entropy

        # 总损失
        loss = policy_loss + 0.5 * value_loss - self.beta * entropy

        # 更新模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _preprocess_state(self, state):
        """将状态转换为适合神经网络的格式"""
        processed = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                processed[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
            else:
                processed[k] = torch.tensor([v], device=self.device)
        return processed

def train():
    env = Tetris2_env()
    input_shape = (1, env.MAPHEIGHT, env.MAPWIDTH)
    agent = PPOAgent(input_shape, num_actions=7)

    batch_size = 128
    episodes = 1000
    max_steps = 200

    for episode in range(episodes):
        print(f"episode = {episode}, training...")
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # 获取动作
            action, prob, value = agent.get_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转换
            agent.store_transition(state, action, prob, value, reward, done)

            state = next_state
            episode_reward += reward
            step_count += 1

            # 定期更新模型
            if len(agent.memory) >= batch_size:
                loss = agent.update(batch_size)
                print(f"Episode {episode}, Step {step_count}, Loss: {loss:.4f}")

        print(f"Episode {episode}, Reward: {episode_reward}, Steps: {step_count}")

        # 定期保存模型
        if (episode + 1) % 100 == 0:
            torch.save(agent.policy.state_dict(), f"tetris_ppo_{episode}.pth")

if __name__ == "__main__":
    print("训练开始")
    train()