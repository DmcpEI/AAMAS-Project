"""
Minimax Q-Learning Hunter
========================

Hunter agent that uses Minimax Q-Learning algorithm to catch prey.
Learns Q-values while assuming prey agents act optimally to evade capture.
"""

import logging
from typing import Tuple, List
from Agents.MinimaxQAgent import MinimaxQAgent

logger = logging.getLogger(__name__)

class MinimaxQHunter(MinimaxQAgent):
    """
    Hunter using Minimax Q-Learning.
    Maximizing player - tries to maximize rewards (catching prey).
    """
    
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.3):
        super().__init__(model, alpha, gamma, epsilon)
        self.epsilon_min = 0.05  # Lower minimum exploration for hunters
        self.epsilon_decay = 0.997  # Slower decay for stable learning
        
    def get_state(self):
        """Get current state: (my_position, prey_positions)"""
        # Get all prey positions (both regular and Q-learning prey)
        prey_positions = tuple(sorted([
            agent.pos for agent in self.model.agents 
            if agent.__class__.__name__.endswith("Prey")
        ]))
        return (self.pos, prey_positions)
    
    def get_other_positions(self, state):
        """Get prey positions from state"""
        return state[1]
    
    def is_maximizing_player(self):
        """Hunters are maximizing players"""
        return True
    
    def calculate_reward(self):
        """Calculate reward based on current situation"""
        reward = -0.1  # Small penalty for each step (time cost)
        
        # Check for prey capture
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        prey_in_cell = [agent for agent in cell_contents 
                       if agent != self and agent.__class__.__name__.endswith("Prey")]
        
        if prey_in_cell:
            reward += 10  # Large reward for catching prey
            logger.info(f"MinimaxQHunter {self.unique_id} caught prey at {self.pos}!")
        
        return reward
    
    def step(self):
        """Execute one step with learning"""
        # Calculate reward from current position
        reward = self.calculate_reward()
        
        # Learn from previous action if we have one
        if hasattr(self, '_previous_prey_action'):
            self.learn_from_experience(reward, self._previous_prey_action)
        
        # Take action using parent's step method
        super().step()
        
        # Observe prey actions for learning (simplified: closest prey)
        prey_positions = self.get_other_positions(self.get_state())
        if prey_positions:
            # For learning, we need to know what the prey did
            # In practice, this would be observed from the environment
            closest_prey_pos = min(prey_positions, 
                                 key=lambda p: self.manhattan_distance(self.pos, p))
            self._previous_prey_action = closest_prey_pos
    
    def get_metrics(self):
        """Extend metrics for MinimaxQHunter"""
        metrics = super().get_metrics()
        metrics.update({
            'agent_type': 'MinimaxQHunter',
            'catches_made': max(0, int(self.total_reward / 10))  # Approximate catches
        })
        return metrics
