"""
data/loader.py - Converts episodes.npy into PyTorch RL training batches
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from generator import DataGenerator

class EpisodeDataset(Dataset):
    def __init__(self, episodes_path="episodes.npy"):

        """data/generator.py will generate episodes.npy at runtime. Load them here"""

        if not os.path.exists(episodes_path):
            raise FileNotFoundError(
                f"{episodes_path} not found. Run data.generator' first."
            )
        self.episodes = np.load(episodes_path)  # Shape: (N_episodes, max_steps, state_dim)
        
        self.n_episodes, self.max_steps, self.state_dim = self.episodes.shape
        print(f"Loaded {self.n_episodes} episodes, {self.max_steps} steps each, {self.state_dim} state dims")
    
    def __len__(self):
        """Number of (state, action, reward, next_state) transitions"""
        return self.n_episodes * (self.max_steps - 1)  # 99 transitions per 100-step episode
    
    def __getitem__(self, idx):
        """Return single RL transition: (state, action, reward, next_state)"""
        episode_idx = idx // (self.max_steps - 1)  # Which episode
        step_idx = idx % (self.max_steps - 1)       # Which step within episode
        
        # Extract transition s_t -> a_t -> r_t -> s_{t+1}
        state = self.episodes[episode_idx, step_idx]           # s_t: full 8D state
        next_state = self.episodes[episode_idx, step_idx + 1]  # s_{t+1}: next full state
        
        # Action was difference between states (thrust vector)
        # Assuming state[2:4] = [vel_x, vel_y], action = delta_vel * dt
        action = next_state[2:4] - state[2:4]  # Simple velocity diff as action proxy
        
        # Reward: negative distance to target + fuel penalty + crash penalty
        # Target assumed at (100,100), crash if dist_to_obstacle < 2.0
        ship_pos = state[:2]
        dist_to_target = np.linalg.norm(ship_pos - np.array([100.0, 100.0]))
        dist_to_obstacle = state[5]  # Assuming state[5] = min dist to obstacles
        fuel = state[4]
        
        reward = -0.1 * dist_to_target  # Progress toward target
        reward -= 0.01 * (100.0 - fuel)  # Fuel efficiency penalty
        if dist_to_obstacle < 2.0:
            reward -= 50.0  # Crash penalty
        if dist_to_target < 5.0:
            reward += 100.0  # Target reached bonus
        
        # Convert to PyTorch tensors (neural net requires this)
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32)
        )

def get_dataloader(batch_size=32, shuffle=True, num_workers=0):
    """Factory function for easy DataLoader creation"""
    dataset = EpisodeDataset()
    print("Checking --- ", dataset.__getitem__(0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Faster GPU transfer
    )


if __name__ == "__main__":
    # Test the loader
    loader = get_dataloader(batch_size=4)
    print("Testing loader...")
    
    for i, batch in enumerate(loader):
        states, actions, rewards, next_states = batch
        print(f"Batch {i}:")
        print(f"  states: {states.shape} (batch, 8)")
        print(f"  actions: {actions.shape} (batch, 2)")
        print(f"  rewards: {rewards.shape} (batch)")
        print(f"  next_states: {next_states.shape} (batch, 8)")
        print(f"  Sample state: {states[0][:5]}...")  # First 5 dims
        if i >= 2:  # Show 3 batches
            break
    print("Loader test complete!")    
