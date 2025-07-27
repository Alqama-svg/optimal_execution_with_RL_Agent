#!/usr/bin/env python
# coding: utf-8

# In[1]:


class MarketSimulator:
    
    def __init__(self, initial_price=100.0):
        self.current_price = initial_price
        self.fundamental_value = initial_price
        self.time_step = 0
        
        # Market parameters
        self.volatility = 0.02
        self.mean_reversion_strength = 0.1
        self.liquidity_depth = 1000  # shares per price level
        
        # Order book state
        self.bid_prices = []
        self.ask_prices = []
        self.bid_volumes = []
        self.ask_volumes = []
        
        self._initialize_order_book()
    
    def _initialize_order_book(self):
        mid_price = self.current_price
        spread = 0.02  # 2 cent spread
        
        # Initialize 5 levels on each side
        for i in range(5):
            bid_price = mid_price - spread/2 - i * 0.01
            ask_price = mid_price + spread/2 + i * 0.01
            
            self.bid_prices.append(bid_price)
            self.ask_prices.append(ask_price)
            self.bid_volumes.append(self.liquidity_depth * (1 - i * 0.1))  # Decreasing volume
            self.ask_volumes.append(self.liquidity_depth * (1 - i * 0.1))
    
    def step(self):
        self.time_step += 1
        
        # Update fundamental value (Ornstein-Uhlenbeck process)
        dt = 1.0
        dW = np.random.normal(0, np.sqrt(dt))
        self.fundamental_value += (
            -self.mean_reversion_strength * (self.fundamental_value - 100.0) * dt +
            self.volatility * dW
        )
        
        # Update current price towards fundamental with noise
        price_change = 0.1 * (self.fundamental_value - self.current_price) + np.random.normal(0, 0.01)
        self.current_price += price_change
        
        # Update order book
        self._update_order_book()
        
        # Add some random market activity
        self._add_market_noise()
    
    def _update_order_book(self):
        mid_price = self.current_price
        spread = 0.02 + abs(np.random.normal(0, 0.005))  # Variable spread
        
        # Update prices
        for i in range(len(self.bid_prices)):
            self.bid_prices[i] = mid_price - spread/2 - i * 0.01
            self.ask_prices[i] = mid_price + spread/2 + i * 0.01
            
            # Add some volume dynamics
            volume_noise = np.random.normal(1.0, 0.1)
            self.bid_volumes[i] = max(100, self.liquidity_depth * (1 - i * 0.1) * volume_noise)
            self.ask_volumes[i] = max(100, self.liquidity_depth * (1 - i * 0.1) * volume_noise)
    
    def _add_market_noise(self):
        if np.random.random() < 0.1:  # 10% chance of market order
            # Random market order consumes some liquidity
            side = np.random.choice(['bid', 'ask'])
            quantity = np.random.randint(50, 200)
            
            if side == 'bid' and self.ask_volumes:
                self.ask_volumes[0] = max(50, self.ask_volumes[0] - quantity)
            elif side == 'ask' and self.bid_volumes:
                self.bid_volumes[0] = max(50, self.bid_volumes[0] - quantity)
    
    def execute_trade(self, quantity: int, side: str = 'buy') -> tuple:
        if quantity <= 0:
            return self.current_price, 0
        
        total_cost = 0
        remaining_quantity = quantity
        depth_consumed = 0
        levels_hit = 0
        
        if side == 'buy':
            # Execute against ask side
            for i, (price, volume) in enumerate(zip(self.ask_prices, self.ask_volumes)):
                if remaining_quantity <= 0:
                    break
                
                trade_quantity = min(remaining_quantity, volume)
                total_cost += price * trade_quantity
                remaining_quantity -= trade_quantity
                levels_hit += 1
                
                # Market impact: reduce available volume
                self.ask_volumes[i] = max(0, self.ask_volumes[i] - trade_quantity)
                
                # If we consumed significant portion of level, count as depth consumed
                if trade_quantity / volume > 0.5:
                    depth_consumed += 1
        
        else:  # sell
            # Execute against bid side
            for i, (price, volume) in enumerate(zip(self.bid_prices, self.bid_volumes)):
                if remaining_quantity <= 0:
                    break
                
                trade_quantity = min(remaining_quantity, volume)
                total_cost += price * trade_quantity
                remaining_quantity -= trade_quantity
                levels_hit += 1
                
                # Market impact: reduce available volume
                self.bid_volumes[i] = max(0, self.bid_volumes[i] - trade_quantity)
                
                if trade_quantity / volume > 0.5:
                    depth_consumed += 1
        
        # Calculate average execution price
        executed_quantity = quantity - remaining_quantity
        avg_execution_price = total_cost / executed_quantity if executed_quantity > 0 else self.current_price
        
        # Market impact on price (temporary)
        impact_factor = min(0.001 * quantity / 1000, 0.01)  # Max 1% impact
        if side == 'buy':
            self.current_price += impact_factor
        else:
            self.current_price -= impact_factor
        
        return avg_execution_price, depth_consumed
    
    def get_market_features(self) -> dict:
        best_bid = self.bid_prices[0] if self.bid_prices else self.current_price - 0.01
        best_ask = self.ask_prices[0] if self.ask_prices else self.current_price + 0.01
        
        # Calculate volume imbalance for multiple levels
        volume_imbalances = []
        for level in range(1, 6):  # 5 levels
            bid_vol = sum(self.bid_volumes[:level]) if len(self.bid_volumes) >= level else 0
            ask_vol = sum(self.ask_volumes[:level]) if len(self.ask_volumes) >= level else 0
            total_vol = bid_vol + ask_vol
            imbalance = bid_vol / total_vol if total_vol > 0 else 0.5
            volume_imbalances.append(imbalance)
        
        return {
            'mid_price': (best_bid + best_ask) / 2,
            'spread': best_ask - best_bid,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'volume_imbalances': volume_imbalances,
            'total_bid_volume': sum(self.bid_volumes),
            'total_ask_volume': sum(self.ask_volumes)
        }

