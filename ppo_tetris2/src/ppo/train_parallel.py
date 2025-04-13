import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch.distributions import Categorical
import torch.nn.functional as F
from tetris2_env import Tetris2_env, decode_action
import matplotlib.pyplot as plt
import os
import copy
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock
import time
from train import PPONetwork

class PPOTrainer:
    def __init__(self, global_policy, optimizer_cls, lr, gamma, clip_epsilon, beta,
                 num_processes, input_shape, num_actions, device):
        self.global_policy = global_policy
        self.optimizer = optimizer_cls(global_policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.num_processes = num_processes
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device

        # Shared objects for synchronization
        self.train_queue = Queue(maxsize=num_processes * 2)
        self.update_counter = Value('i', 0)
        self.lock = Lock()

    def train(self, num_episodes, batch_size, max_steps, checkpoint_interval):
        processes = []

        # Start worker processes
        for worker_id in range(self.num_processes):
            p = Process(target=self.worker,
                        args=(worker_id, num_episodes, batch_size, max_steps))
            p.start()
            processes.append(p)
            time.sleep(0.1)  # Stagger process start

        # Training loop in main process
        reward_log = []
        loss_log = []
        episode_counter = 0

        try:
            while episode_counter < num_episodes:
                # Collect experiences from workers
                while not self.train_queue.empty() and episode_counter < num_episodes:
                    episode_reward, batch = self.train_queue.get()
                    reward_log.append(episode_reward)
                    episode_counter += 1

                    # Print training info
                    avg_reward = np.mean(reward_log[-10:]) if len(reward_log) >= 10 else np.mean(reward_log)
                    print(f"Episode {episode_counter}: Reward {episode_reward:.1f}, Avg Reward {avg_reward:.1f}")

                    # Update policy if we have enough samples
                    if len(batch) >= batch_size:
                        loss = self.update_policy(batch)
                        loss_log.append(loss)
                        print(f"Update {self.update_counter.value}: Loss {loss:.4f}")

                    # Save checkpoint periodically
                    if episode_counter % checkpoint_interval == 0:
                        self.save_checkpoint(episode_counter, reward_log, loss_log)

        except KeyboardInterrupt:
            print("Training interrupted, cleaning up...")

        finally:
            # Clean up
            for p in processes:
                p.terminate()
                p.join()

            # Save final model
            self.save_checkpoint(episode_counter, reward_log, loss_log, final=True)

        return reward_log, loss_log

    def worker(self, worker_id, num_episodes, batch_size, max_steps):
        # Create local environment and policy
        env = Tetris2_env()
        local_policy = PPONetwork(self.input_shape, self.num_actions).to(self.device)
        local_policy.train()

        # Sync with global policy
        local_policy.load_state_dict(self.global_policy.state_dict())

        episode_count = 0
        memory = []

        while episode_count < num_episodes // self.num_processes:
            # Collect episode
            state = env.reset()
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < max_steps:
                # Get action from local policy
                action, prob, value = self.get_action(local_policy, state, env)

                # Take step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Store transition
                memory.append((state, action, prob, value, reward, done))

                state = next_state
                step_count += 1

                # Sync with global policy periodically
                if step_count % 50 == 0:
                    with self.lock:
                        local_policy.load_state_dict(self.global_policy.state_dict())

                # Send batch to main process if ready
                if len(memory) >= batch_size:
                    with self.lock:
                        self.train_queue.put((episode_reward, memory))
                        memory = []
                        episode_count += 1
                        break

            # If episode ended before filling batch
            if len(memory) > 0 and (done or step_count >= max_steps):
                with self.lock:
                    self.train_queue.put((episode_reward, memory))
                    memory = []
                    episode_count += 1

    def get_action(self, policy, state, env):
        state = self._preprocess_state(state)

        width = state["current_grid"].shape[-1]
        height = state["current_grid"].shape[-2]
        rotation_size = 4
        total_actions = width * height * rotation_size

        valid_actions = env.get_valid_actions()

        with torch.no_grad():
            placement_logits, next_block_logits, value = policy(state)
            placement_logits = placement_logits.view(-1)

            placement_mask = torch.full_like(placement_logits, float('-inf'))
            if valid_actions:
                for idx in valid_actions:
                    placement_mask[idx] = placement_logits[idx]
            else:
                placement_mask = placement_logits

            placement_probs = torch.softmax(placement_mask, dim=0)
            placement_dist = Categorical(placement_probs)
            placement_action_flat = placement_dist.sample()
            x, y, o = decode_action(placement_action_flat.item(), width)
            placement_action = (x, y, o)

            legal_next_blocks = []
            piece_count = env.typeCountForColor
            opponent_color = 1 - env.currBotColor

            for next_block in range(7):
                temp_piece_count = copy.deepcopy(piece_count[opponent_color])
                temp_piece_count[next_block] += 1

                if max(temp_piece_count) - min(temp_piece_count) <= 2:
                    legal_next_blocks.append(next_block)

            next_block_logits = next_block_logits.view(-1)
            next_block_mask = torch.full_like(next_block_logits, float('-inf'))
            if legal_next_blocks:
                for idx in legal_next_blocks:
                    next_block_mask[idx] = next_block_logits[idx]
            else:
                next_block_mask = next_block_logits

            next_block_probs = torch.softmax(next_block_mask, dim=-1)
            next_block_dist = Categorical(next_block_probs)
            next_block_action = next_block_dist.sample()

            action_prob = placement_dist.log_prob(placement_action_flat).exp() * \
                          next_block_dist.log_prob(next_block_action).exp()

            return (placement_action, next_block_action.item()), action_prob.item(), value.item()

    def update_policy(self, batch):
        states, actions, old_probs, old_values, rewards, dones = zip(*batch)

        # Preprocess batch
        states = [self._preprocess_state(s) for s in states]
        batched_states = self._batch_states(states)

        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
            advantages.insert(0, R - old_values[i])

        advantages = torch.tensor(advantages, device=self.device).float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(returns, device=self.device).float()
        old_probs = torch.tensor(old_probs, device=self.device).float()
        old_values = torch.tensor(old_values, device=self.device).float()

        # Get action tensors
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

        # Forward pass
        placement_logits, next_block_logits, values = self.global_policy(batched_states)

        placement_probs = torch.softmax(placement_logits.view(len(batch), -1), dim=-1)
        placement_dist = Categorical(placement_probs)
        placement_log_probs = placement_dist.log_prob(placement_actions)

        next_block_probs = torch.softmax(next_block_logits, dim=-1)
        next_block_dist = Categorical(next_block_probs)
        next_block_log_probs = next_block_dist.log_prob(next_block_actions)

        new_probs = (placement_log_probs + next_block_log_probs).exp()
        ratios = new_probs / old_probs

        # PPO loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Entropy bonus
        entropy = placement_dist.entropy().mean() + next_block_dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - self.beta * entropy

        # Update global policy
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.global_policy.parameters(), 0.5)

        self.optimizer.step()

        with self.update_counter.get_lock():
            self.update_counter.value += 1

        return loss.item()

    def _preprocess_state(self, state):
        """Convert single state dict to tensor format"""
        processed = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                processed[k] = torch.FloatTensor(v).unsqueeze(0).to(self.device)
            else:
                processed[k] = torch.tensor([v], device=self.device)
        return processed

    def _batch_states(self, state_list):
        """Combine multiple preprocessed state dicts into batched dict"""
        batch = {}
        for key in state_list[0]:
            batch[key] = torch.cat([s[key] for s in state_list], dim=0)
        return batch

    def save_checkpoint(self, episode, reward_log, loss_log, final=False):
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.global_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_log': reward_log,
            'loss_log': loss_log
        }

        if final:
            torch.save(checkpoint, "checkpoint/ppo_checkpoint_final.pth")
            torch.save(self.global_policy.state_dict(), "model/tetris_ppo_final.pth")
        else:
            torch.save(checkpoint, f"checkpoint/ppo_checkpoint_{episode}.pth")
            torch.save(self.global_policy.state_dict(), f"model/tetris_ppo_{episode}.pth")

        # Plot training curves
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(reward_log, label="Episode Reward")
        if len(reward_log) >= 10:
            avg_reward = [np.mean(reward_log[max(0, i-10):i+1]) for i in range(len(reward_log))]
            plt.plot(avg_reward, label="Avg Reward (10 eps)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(loss_log, label="Loss", color='orange')
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"pic/training_curve_{episode}.png" if not final else "pic/training_curve_final.png")
        plt.close()

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Tetris2_env()
    input_shape = (1, env.MAPHEIGHT, env.MAPWIDTH)
    num_actions = 7

    # Create global policy
    global_policy = PPONetwork(input_shape, num_actions).to(device)
    global_policy.share_memory()  # For multiprocessing

    # Training parameters
    num_processes = mp.cpu_count() - 1  # Leave one core free
    print(f"Using {num_processes} processes for training")

    trainer = PPOTrainer(
        global_policy=global_policy,
        optimizer_cls=optim.Adam,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        beta=0.01,
        num_processes=num_processes,
        input_shape=input_shape,
        num_actions=num_actions,
        device=device
    )

    # Start training
    reward_log, loss_log = trainer.train(
        num_episodes=10000,
        batch_size=128,
        max_steps=200,
        checkpoint_interval=100
    )

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn')
    main()