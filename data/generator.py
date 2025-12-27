# data/generator.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from typing import List, Dict, Tuple
from config import MAX_STEPS, FUEL_START, WORLD_SIZE, DT


class DataGenerator:
    """
    Generates synthetic training data for the spaceship RL agent to train itself on. 
    It's almost like the ship is seeing the life of another ship and learning from it. 
    """
    
    def __init__(self):
        """Initialize with world parameters."""
        self.world_size = WORLD_SIZE  # 100x100 grid boundaries
        self.dt = DT        # Timestep for physics
    
    def spawn_obstacles(self, n_planets=3, n_debris=10, n_blackholes=1) -> List[Dict]:
        """
        Generate random obstacles for one episode.
        
        Args:
            n_planets: Number of planets (r=5-10)
            n_debris: Number of debris pieces (r=0.5-1)
            n_blackholes: Number of black holes (r=2, instant death)
            
        Returns:
            List of obstacle dicts: [{"type": str, "pos": [float,float], "radius": float}]
        """
        obstacles = []
        
        # Planets (large, slow)
        for _ in range(n_planets):
            pos = [np.random.uniform(20, self.world_size-20), 
                   np.random.uniform(20, self.world_size-20)]
            radius = np.random.uniform(5, 10)
            obstacles.append({"type": "planet", "pos": pos, "radius": radius})
        
        # Debris (small, fast)
        for _ in range(n_debris):
            pos = [np.random.uniform(0, self.world_size), 
                   np.random.uniform(0, self.world_size)]
            radius = np.random.uniform(0.5, 1.0)
            obstacles.append({"type": "debris", "pos": pos, "radius": radius})
        
        # Black holes (deadly)
        for _ in range(n_blackholes):
            pos = [np.random.uniform(30, self.world_size-30), 
                   np.random.uniform(30, self.world_size-30)]
            radius = 2.0
            obstacles.append({"type": "blackhole", "pos": pos, "radius": radius})
        
        return obstacles
    
    def compute_state_vector(self, ship_pos: np.ndarray, ship_vel: np.ndarray, 
                           fuel: float, obstacles: List[Dict]) -> np.ndarray:
        """
        Compute 8-element state vector from current situation.
        
        Args:
            ship_pos: [x, y]
            ship_vel: [vx, vy] 
            fuel: Current fuel level
            obstacles: List from spawn_obstacles()
            
        Returns:
            np.array([ship_x, ship_y, vel_x, vel_y, fuel, min_dist, obj_angle, terminal]
        """
        
        ship_x, ship_y = ship_pos
    
        vel_x, vel_y = ship_vel
        
        
        # Find nearest obstacle
        nearest_dist = float('inf')
        # Angle from ship to nearest obstacle
        obj_angle = 0.0
        
        for obs in obstacles:
            # Compute spatial displacement
            dx = obs["pos"][0] - ship_x
            dy = obs["pos"][1] - ship_y
            # Use Euclidean distance to compute distance
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < nearest_dist:
                nearest_dist = dist
                obj_angle = np.arctan2(dy, dx)
        
        # Boundary check (wrap around edges)
        if ship_x < 0: ship_x = self.world_size
        if ship_x > self.world_size: ship_x = 0
        if ship_y < 0: ship_y = self.world_size  
        if ship_y > self.world_size: ship_y = 0
        
        
        terminal = 1.0 if nearest_dist < 1.0 or fuel <= 0 else 0.0
        return np.array([ship_x, ship_y, vel_x, vel_y, 
                fuel, nearest_dist, obj_angle, terminal], dtype=np.float32)


    
    def generate_single_episode(self, max_steps: int = 100) -> np.ndarray:
        """
        Simulate one complete episode with random policy.
        
        Returns:
            np.array(shape=[steps, 7]) - trajectory until termination
        """
        # Initial state
        ship_pos = np.array([0.0, 0.0])
        ship_vel = np.array([0.0, 0.0])
        fuel = FUEL_START
        obstacles = self.spawn_obstacles()
        
        episode = []
        
        for step in range(max_steps):
            # Random action (thrust 0-1, angle -pi to pi)
            thrust = np.random.uniform(0, 0.5)
            angle = np.random.uniform(-np.pi, np.pi)
            # X and Y thrust components: 
            action = np.array([thrust * np.cos(angle), thrust * np.sin(angle)])
            
            # Simple physics (no env.step() - pure random policy)
            ship_vel += action * self.dt
            ship_pos += ship_vel * self.dt
            # Wrap around world edges
            ship_pos[0] = ship_pos[0] % self.world_size
            ship_pos[1] = ship_pos[1] % self.world_size

            fuel -= np.linalg.norm(action) * 0.1
            
            # Check termination (same logic as env)
            nearest_dist = min([np.linalg.norm(ship_pos - np.array(obs["pos"])) 
                              for obs in obstacles])
            
            # Compute terminal before state recording
            terminal = 1.0 if nearest_dist < 1.0 or fuel <= 0 else 0.0
            if terminal == 1.0:
                break

            # Record state 
            state = self.compute_state_vector(ship_pos, ship_vel, fuel, obstacles)
            episode.append(state)
        
        # Pad to max_steps if short
        while len(episode) < max_steps:
            episode.append(np.zeros(8))
        
        return np.array(episode, dtype=np.float32)
    
    def generate_episodes(self, n_episodes: int = 10000) -> np.ndarray:
        """
        MAIN FUNCTION - Generate full training dataset.
        
        Returns:
            np.array(shape=[n_episodes, max_steps, 8])
        """
        print(f"Generating {n_episodes} synthetic episodes...")
        episodes = []
        
        for i in range(n_episodes):
            if i % 1000 == 0:
                print(f"Progress: {i}/{n_episodes}")
            
            episode = self.generate_single_episode()
            episodes.append(episode)
        
        episodes_array = np.stack(episodes, axis=0)  # [n_episodes, max_steps, 7]
        print(f"Generated shape: {episodes_array.shape}")
        return episodes_array

if __name__ == "__main__":
    gen = DataGenerator()
    episodes = gen.generate_episodes(n_episodes=10)  # Small test
    print(f"Generated shape: {episodes.shape}")