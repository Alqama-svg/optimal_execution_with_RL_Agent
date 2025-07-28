#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_aggressive_strategy(env):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = 2  # Always buy 2 × Qmin
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

