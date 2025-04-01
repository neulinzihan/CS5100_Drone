import numpy as np
import torch

from multi_drone_env import MultiDroneCoverageEnv
from agent import DroneDQNAgent

def train_multi_drone(n_episodes=1000, 
                      grid_size=10,
                      num_drones=3,
                      coverage_radii=None,
                      max_steps=100):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = MultiDroneCoverageEnv(grid_size=grid_size,
                                num_drones=num_drones,
                                coverage_radii=coverage_radii,
                                max_steps=max_steps,
                                normalize_rewards=True)  # local rewards
    
    # Create one DQN agent per drone
    agents = []
    for i in range(num_drones):
        agent = DroneDQNAgent(obs_size=3, action_size=5, device=device)
        agents.append(agent)
    
    all_rewards = []  # We'll track the sum of local rewards for each episode
    
    for ep in range(n_episodes):
        obs_dict = env.reset()  # {drone_id: array([row, col, coverage_frac])}
        done = False
        ep_reward = np.zeros(num_drones, dtype=np.float32)
        
        for t in range(max_steps):
            action_dict = {}
            for i in range(num_drones):
                action_dict[i] = agents[i].act(obs_dict[i])
            
            next_obs_dict, reward_dict, done, _ = env.step(action_dict)
            
            # Store transitions and accumulate reward
            for i in range(num_drones):
                agents[i].step(obs_dict[i],
                               action_dict[i],
                               reward_dict[i],
                               next_obs_dict[i],
                               float(done))
                ep_reward[i] += reward_dict[i]
            
            # Make next observations current
            obs_dict = next_obs_dict
            
            # Learn
            for i in range(num_drones):
                agents[i].learn()
            
            if done:
                break
        
        avg_rew = np.mean(ep_reward)  # average local reward across drones
        all_rewards.append(avg_rew)
        
        if (ep+1) % 50 == 0:
            print(f"Episode {ep+1}/{n_episodes}, avg local reward={avg_rew:.4f}")
    
    env.close()
    return agents, all_rewards

def demo_run(agents, grid_size=10, coverage_radii=None, max_steps=50):
    """
    Run a single test episode with a set of trained agents.
    We'll print the environment's textual rendering each step.
    """
    env = MultiDroneCoverageEnv(
        grid_size=grid_size,
        num_drones=len(agents),
        coverage_radii=coverage_radii,
        max_steps=max_steps,
        normalize_rewards=True
    )
    obs_dict = env.reset()
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        action_dict = {}
        for i, agent in enumerate(agents):
            a = agent.act(obs_dict[i])  # This is still epsilon-greedy
            action_dict[i] = a
        
        next_obs_dict, reward_dict, done, _ = env.step(action_dict)
        
        step_count += 1
        env.render()
        
        # Print each drone's local reward
        print("Step:", step_count, "Rewards:", reward_dict)
        
        obs_dict = next_obs_dict
    
    env.close()

if __name__=="__main__":
    # Example usage
    trained_agents, rewards = train_multi_drone(
        n_episodes=500, 
        grid_size=10,
        num_drones=2,
        coverage_radii=[1, 2],
        max_steps=50
    )
    
    print("Training complete.  Let's run a short demo episode:")
    demo_run(trained_agents, grid_size=10, coverage_radii=[1,2], max_steps=10)
