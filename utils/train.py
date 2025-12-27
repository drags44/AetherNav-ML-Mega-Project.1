from data.loader import get_dataloader
from agents.rl import PPOAgent

loader = get_dataloader(batch_size=32)  # Automatic EpisodeDataset + DataLoader
agent = PPOAgent()

for epoch in range(100):
    for states, actions, rewards, next_states in loader:
        agent.update(states, actions, rewards, next_states)

