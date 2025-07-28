#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_twap_strategy(env, order_size=MAX_ORDER_SIZE, duration=MAX_TIME_SECONDS):
    state = env.reset()
    steps = duration
    shares_per_step = max(1, order_size // steps)
    total_reward = 0
    
    for step in range(steps):
        if env.inventory <= 0:
            break
        
        # Convert shares to action (ensuring we don't exceed available actions)
        action_size = min(shares_per_step, env.inventory)
        action = min(4, max(1, action_size // Q_MIN))  # Convert to action space
        
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

