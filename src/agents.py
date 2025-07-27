#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def evaluate_twap_policy(OptimalExecEnv):
    obs, _ = env.reset()
    total_reward = 0
    
    # Calculate TWAP rate
    shares_per_step = env.total_shares / env.total_time_steps
    
    while True:
        # Determine TWAP action
        if shares_per_step >= env.q_min:
            action = min(4, int(shares_per_step / env.q_min))
        else:
            action = 1 if env.shares_remaining > 0 else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward

def evaluate_aggressive_policy(env: OptimalExecEnv) -> float:
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        action = 2  # Always execute 2*q_min
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward

def evaluate_passive_policy(env: OptimalExecEnv) -> float:
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        if np.random.random() < 0.6:
            action = 0  # Do nothing
        else:
            action = np.random.randint(1, 5)  # Random execution
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return total_reward

