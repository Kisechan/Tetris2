# train_parallel.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from typing import Dict, List, Tuple
import gym
from tetris2_gym import Tetris2GymEnv, register_env
from tetris2_env import Tetris2_env, decode_action
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

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

class PPOParallelAgent:
    def __init__(self, input_shape, num_actions, n_envs=4, lr=3e-4, gamma=0.99, clip_epsilon=0.2, beta=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.n_envs = n_envs
        self.memory = deque(maxlen=10000)

        # 用于存储每个环境的中间状态
        self.env_states = [None] * n_envs
        self.env_actions = [None] * n_envs
        self.env_probs = [None] * n_envs
        self.env_values = [None] * n_envs

    def get_actions(self, states: List[Dict], envs=None) -> Tuple[List[Tuple], List[float], List[float]]:
        """批量获取动作(不依赖底层环境实例)"""
        processed_states = [self._preprocess_state(s) for s in states]
        batched_states = self._batch_states(processed_states)

        with torch.no_grad():
            placement_logits, next_block_logits, values = self.policy(batched_states)
            placement_logits = placement_logits.view(len(states), -1)
            next_block_logits = next_block_logits.view(len(states), -1)
            values = values.view(-1).cpu().numpy().tolist()

            actions = []
            probs = []

            for i, state in enumerate(states):
                width = state["current_grid"].shape[-1]

                # 处理放置动作 - 假设所有动作都合法（实际项目中需要验证）
                placement_probs = torch.softmax(placement_logits[i], dim=0)
                placement_dist = Categorical(placement_probs)
                placement_action_flat = placement_dist.sample()
                x, y, o = decode_action(placement_action_flat.item(), width)
                placement_action = (x, y, o)

                # 处理下一个方块选择 - 简化版
                next_block_probs = torch.softmax(next_block_logits[i], dim=-1)
                next_block_dist = Categorical(next_block_probs)
                next_block_action = next_block_dist.sample()

                action_prob = placement_dist.log_prob(placement_action_flat).exp() * \
                              next_block_dist.log_prob(next_block_action).exp()

                actions.append((placement_action, next_block_action.item()))
                probs.append(action_prob.item())

                # 保存中间状态
                self.env_states[i] = state
                self.env_actions[i] = actions[-1]
                self.env_probs[i] = probs[-1]
                self.env_values[i] = values[i]

            return actions, probs, values

    def store_transitions(self, rewards: List[float], dones: List[bool]):
        """存储转换到记忆缓冲区"""
        for i in range(len(rewards)):
            if self.env_states[i] is not None:
                self.memory.append((
                    self.env_states[i],
                    self.env_actions[i],
                    self.env_probs[i],
                    self.env_values[i],
                    rewards[i],
                    dones[i]
                ))
                # 重置环境状态
                self.env_states[i] = None

    def update(self, batch_size):
        """更新策略"""
        if len(self.memory) < batch_size:
            return None

        # 从内存中采样
        samples = random.sample(self.memory, batch_size)
        states, actions, old_probs, old_values, rewards, dones = zip(*samples)

        # 预处理状态
        states = [self._preprocess_state(s) for s in states]
        batched_states = self._batch_states(states)

        # 计算折扣回报和优势
        returns = []
        advantages = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
            advantages.insert(0, R - old_values[i])

        # 转换为张量
        advantages = torch.tensor(advantages, device=self.device).float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, device=self.device).float()
        old_probs = torch.tensor(old_probs, device=self.device).float()
        old_values = torch.tensor(old_values, device=self.device).float()

        # 获取动作张量
        width = batched_states["current_grid"].shape[-1]
        height = batched_states["current_grid"].shape[-2]

        placement_actions = []
        next_block_actions = []
        for action in actions:
            placement, next_block = action
            placement_flat = (placement[1]-1)*width*4 + (placement[0]-1)*4 + placement[2]
            placement_actions.append(placement_flat)
            next_block_actions.append(next_block)

        placement_actions = torch.tensor(placement_actions, device=self.device).long()
        next_block_actions = torch.tensor(next_block_actions, device=self.device).long()

        # 计算新策略的概率和价值
        placement_logits, next_block_logits, values = self.policy(batched_states)

        placement_probs = torch.softmax(placement_logits.view(batch_size, -1), dim=-1)
        placement_dist = Categorical(placement_probs)
        placement_log_probs = placement_dist.log_prob(placement_actions)

        next_block_probs = torch.softmax(next_block_logits, dim=-1)
        next_block_dist = Categorical(next_block_probs)
        next_block_log_probs = next_block_dist.log_prob(next_block_actions)

        new_probs = (placement_log_probs + next_block_log_probs).exp()
        ratios = new_probs / old_probs

        # PPO目标函数
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)

        # 熵奖励
        entropy = placement_dist.entropy().mean() + next_block_dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - self.beta * entropy

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def _get_legal_next_blocks(self, env):
        """获取合法的下一个方块选择"""
        legal_next_blocks = []
        piece_count = env.typeCountForColor
        opponent_color = 1 - env.currBotColor

        for next_block in range(7):
            temp_piece_count = piece_count[opponent_color].copy()
            temp_piece_count[next_block] += 1
            if max(temp_piece_count) - min(temp_piece_count) <= 2:
                legal_next_blocks.append(next_block)
        return legal_next_blocks

    def _preprocess_state(self, state):
        """处理可能被VecEnv包装过的状态"""
        if isinstance(state, str):
            raise ValueError(f"Unexpected string state: {state}")

        # 如果是从VecEnv返回的单个状态
        if not isinstance(state, dict):
            try:
                state = state[0]  # 尝试解包
            except (TypeError, IndexError):
                pass

        # 确保最终得到字典
        if not isinstance(state, dict):
            raise TypeError(f"Expected dict state, got {type(state)}: {state}")

        processed = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                processed[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
            else:
                processed[k] = torch.tensor([v], device=self.device)
        return processed

    def _batch_states(self, state_list):
        """将多个预处理后的状态 dict 合并为 batched dict"""
        batch = {}
        for key in state_list[0]:
            batch[key] = torch.cat([s[key] for s in state_list], dim=0)
        return batch

def make_env(env_id, rank=0, seed=0):
    def _init():
        # 子进程中需要重新注册
        from tetris2_gym import register_env
        register_env()

        env = gym.make(env_id)
        # env.seed(seed + rank)
        return env
    return _init

def make_parallel_envs(env_id, n_envs=4, parallel=True):
    if parallel:
        return SubprocVecEnv([make_env(env_id, i) for i in range(n_envs)])
    else:
        return DummyVecEnv([make_env(env_id, i) for i in range(n_envs)])

def train_parallel():
    # Windows多进程特殊处理
    mp.set_start_method('spawn', force=True)
    register_env()

    # 先尝试使用DummyVecEnv
    # print("Testing with DummyVecEnv first...")
    # try:
    #     n_envs = 1
    #     envs = make_parallel_envs('Tetris2-v0', n_envs=n_envs, parallel=False)
    #     print("DummyVecEnv test passed!")
    #     envs.close()
    # except Exception as e:
    #     print(f"DummyVecEnv test failed: {e}")
    #     return

    # 然后尝试真正的并行
    print("Attempting parallel training...")
    try:
        n_envs = 4
        envs = make_parallel_envs('Tetris2-v0', n_envs=n_envs, parallel=True)

        input_shape = (1, 20, 10)
        agent = PPOParallelAgent(input_shape, num_actions=7, n_envs=n_envs)

        mp.freeze_support()

        # 加载检查点
        checkpoint_path = "checkpoint/ppo_checkpoint_latest.pth"
        reward_log = []
        start_episode = 0

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            reward_log = checkpoint['reward_log']
            start_episode = checkpoint['episode'] + 1
            print(f"Loaded checkpoint, resuming from episode {start_episode}")

        # 训练参数
        batch_size = 128
        episodes = 10000
        max_steps = 200
        save_interval = 100

        try:
            for episode in range(start_episode, episodes):
                states = envs.reset()
                print(f"Env reset states type: {type(states)}")  # 应该是list或特殊类型
                episode_rewards = [0] * n_envs
                dones = [False] * n_envs

                for step in range(max_steps):
                    # 修改后的调用方式
                    actions, probs, values = agent.get_actions(states)

                    # VecEnv会自动处理动作列表
                    next_states, rewards, dones, infos = envs.step(actions)

                    # 存储转换
                    agent.store_transitions(rewards, dones)

                    # 更新状态和奖励
                    states = next_states
                    for i in range(n_envs):
                        episode_rewards[i] += rewards[i]

                    # 检查是否有环境结束
                    if any(dones):
                        for i, done in enumerate(dones):
                            if done:
                                print(f"Env {i} finished: Reward {episode_rewards[i]:.2f}")
                                reward_log.append(episode_rewards[i])
                                episode_rewards[i] = 0

                    # 定期更新策略
                    if len(agent.memory) >= batch_size:
                        loss = agent.update(batch_size)
                        print(f"Training Step: Loss {loss:.4f}")

                # 保存检查点
                if (episode + 1) % save_interval == 0:
                    torch.save({
                        'episode': episode,
                        'model_state_dict': agent.policy.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'reward_log': reward_log
                    }, f"checkpoint/ppo_checkpoint_{episode + 1}.pth")

                    # 绘制训练曲线
                    plt.figure()
                    plt.plot(reward_log, label="Reward")
                    plt.xlabel("Episode")
                    plt.ylabel("Reward")
                    plt.title("Training Reward Curve")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f"pic/reward_curve_{episode + 1}.png")
                    plt.close()

        finally:
            envs.close()
    except Exception as e:
        print(f"Parallel training failed: {e}")
    finally:
        if 'envs' in locals():
            envs.close()

if __name__ == "__main__":
    print("训练开始")
    train_parallel()