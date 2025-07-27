#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class OptimalExecEnv(gym.Env):
    def __init__(self, total_shares=20000, total_time_steps=1800, q_min=20):
        super(OptimalExecEnv, self).__init__()
        
        # Environment parameters (from paper)
        self.total_shares = total_shares
        self.total_time_steps = total_time_steps
        self.q_min = q_min
        self.alpha = 2.0  # Depth penalty coefficient (was missing!)
        self.end_of_time_penalty = 5.0  # Penalty per unexecuted share (was missing!)
        
        # Action space: 5 discrete actions [0, 1, 2, 3, 4]
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [% holdings, % time, 5x volume_imbalance, best_bid, best_ask, spread, market_features]
        # Total: 12 features
        self.observation_space = spaces.Box(
            low=np.array([0.0] * 12, dtype=np.float32),
            high=np.array([1.0] * 7 + [200.0] * 5, dtype=np.float32),  # Prices can be > 1
            dtype=np.float32
        )
        
        # Initialize market simulator (was missing!)
        self.market_simulator = MarketSimulator()
        
        # State variables
        self.current_step = 0
        self.shares_remaining = 0
        self.arrival_price = 0.0
        self.total_executed_value = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset internal state
        self.current_step = 0
        self.shares_remaining = self.total_shares
        self.total_executed_value = 0.0
        
        # Reset market simulator
        self.market_simulator = MarketSimulator()
        self.arrival_price = self._get_market_price()  # Now implemented!
        
        # Return initial observation
        obs = self._get_obs()  # Now implemented!
        return obs, {}
    
    def step(self, action):
        # 1. Determine quantity to trade (was missing!)
        quantity_to_trade = self._get_quantity_from_action(action)
        
        # 2. Execute trade using market simulator (now implemented!)
        execution_price, depth_consumed = self.market_simulator.execute_trade(
            quantity_to_trade, side='buy'
        )
        
        # 3. Calculate reward using the paper's formula
        if quantity_to_trade > 0:
            implementation_shortfall = quantity_to_trade * (self.arrival_price - execution_price)
            impact_penalty = self.alpha * depth_consumed
            reward = implementation_shortfall - impact_penalty
            
            # Update state
            self.shares_remaining -= quantity_to_trade
            self.total_executed_value += execution_price * quantity_to_trade
        else:
            reward = 0
        
        # 4. Advance time
        self.current_step += 1
        self.market_simulator.step()
        
        # 5. Check termination
        terminated = (self.shares_remaining <= 0) or (self.current_step >= self.total_time_steps)
        
        # 6. Apply end-of-period penalty
        if terminated and self.shares_remaining > 0:
            reward -= self.end_of_time_penalty * self.shares_remaining
        
        # 7. Get next observation
        obs = self._get_obs()
        
        # 8. Prepare info dict
        info = {
            'shares_remaining': self.shares_remaining,
            'execution_progress': 1.0 - (self.shares_remaining / self.total_shares),
            'time_progress': self.current_step / self.total_time_steps,
            'avg_execution_price': (self.total_executed_value / 
                                  (self.total_shares - self.shares_remaining)) 
                                  if self.shares_remaining < self.total_shares else self.arrival_price
        }
        
        return obs, reward, terminated, False, info
    
    def _get_obs(self) -> np.ndarray:
        # Get market features from simulator
        market_features = self.market_simulator.get_market_features()
        
        # Calculate state components
        holdings_pct = self.shares_remaining / self.total_shares
        time_pct = self.current_step / self.total_time_steps
        
        # Volume imbalances (5 levels)
        volume_imbalances = market_features['volume_imbalances']
        
        # Price features (normalized)
        best_bid_norm = market_features['best_bid'] / 100.0
        best_ask_norm = market_features['best_ask'] / 100.0
        spread_norm = market_features['spread'] / 1.0  # Normalize by typical spread
        mid_price_norm = market_features['mid_price'] / 100.0
        
        # Additional market features
        total_liquidity = (market_features['total_bid_volume'] + 
                          market_features['total_ask_volume']) / 10000.0  # Normalize
        
        # Combine all features (total: 12 features)
        obs = np.array([
            holdings_pct,           # 1
            time_pct,              # 2
            *volume_imbalances,    # 3-7 (5 levels)
            best_bid_norm,         # 8
            best_ask_norm,         # 9
            spread_norm,           # 10
            mid_price_norm,        # 11
            total_liquidity        # 12
        ], dtype=np.float32)
        
        return obs
    
    def _get_quantity_from_action(self, action: int) -> int:
        if action == 0:
            return 0
        else:
            quantity = action * self.q_min
            return min(quantity, self.shares_remaining)  # Don't exceed remaining shares
    
    def _get_market_price(self) -> float:
        market_features = self.market_simulator.get_market_features()
        return market_features['mid_price']

