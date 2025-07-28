#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def run_random_strategy(env):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = random.randint(0, 4)
        _, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

