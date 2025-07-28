#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_passive_strategy(env):
    state = env.reset()
    total_reward = 0
    
    while True:
        if random.random() < 0.6:
            action = 0  # do nothing
        else:
            action = random.randint(1, 4)
        
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

