def plot_single_episode(episode_idx=0, save_path="episode_0.png"):
  episode = np.load("data/episodes.npy")[episode_idx]  # (100, 8)
  ship_traj = episode[:, :2]                           # ship_x, ship_y
  obstacles = extract_obstacles(episode)               # Parse obstacle positions
  plt.plot(ship_traj[:,0], ship_traj[:,1], "b-", label="Ship path")
  plt.scatter(obstacles[:,0], obstacles[:,1], "ro", label="Obstacles")
  plt.savefig(save_path)

def plot_state_distributions():
  episodes = np.load("data/episodes.npy")
  states = episodes.reshape(-1, 8)  # All states flattened
  plt.hist(states[:,0], bins=50, alpha=0.7, label="ship_x")
  plt.hist(states[:,4], bins=50, alpha=0.7, label="fuel")
  plt.legend()
  plt.savefig("state_dist.png")

if __name__ == "__main__":
  plot_single_episode(0)
  plot_state_distributions()
