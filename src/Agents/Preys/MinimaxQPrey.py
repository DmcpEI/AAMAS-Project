"""
Minimax Q-Learning Prey
======================

Prey agent that uses Minimax Q-Learning algorithm to evade hunters.
Learns Q-values while assuming hunter agents act optimally to catch prey.
"""

import logging
from typing import Tuple, List
from Agents.MinimaxQAgent import MinimaxQAgent
from Models.ModelConfig import ModelConfig

logger = logging.getLogger(__name__)

class MinimaxQPrey(MinimaxQAgent):
    """
    Prey using Minimax Q-Learning.
    Minimizing player - tries to minimize penalties (being caught).
    """
    
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.3):
        super().__init__(model, alpha, gamma, epsilon)
        self.epsilon_min = 0.1  # Higher minimum exploration for prey
        self.epsilon_decay = 0.998  # Slower decay to maintain exploration
        self._is_dead = False
        
    def get_state(self):
        """Get current state: (my_position, hunter_positions)"""
        # Get all hunter positions
        hunter_positions = tuple(sorted([
            agent.pos for agent in self.model.agents 
            if agent.__class__.__name__.endswith("Hunter")
        ]))
        return (self.pos, hunter_positions)
    
    def get_other_positions(self, state):
        """Get hunter positions from state"""
        return state[1]
    
    def is_maximizing_player(self):
        """Prey are minimizing players"""
        return False
    
    def calculate_reward(self):
        """Calculate reward using updated reward structure."""
        # Check if caught by hunter
        cell_contents = self.model.grid.get_cell_list_contents([self.pos])
        hunters_in_cell = [agent for agent in cell_contents 
                        if agent != self and agent.__class__.__name__.endswith("Hunter")]
        
        if hunters_in_cell:
            self._is_dead = True
            return ModelConfig.DEATH_PENALTY  # -10
        else:
            return ModelConfig.SURVIVAL_REWARD  # +1.0
    
    def step(self):
        """Execute one step with learning"""
        if self._is_dead:
            logger.debug(f"MinimaxQPrey {self.unique_id} skipping step (dead)")
            return
        
        # Calculate reward from current position
        reward = self.calculate_reward()
        
        # Learn from previous action if we have one
        if hasattr(self, '_previous_hunter_action'):
            self.learn_from_experience(reward, self._previous_hunter_action)
        
        # If caught, handle death
        if self._is_dead:
            self.die()
            return
        
        # Take action using parent's step method
        super().step()
        
        # Observe hunter actions for learning (simplified: closest hunter)
        hunter_positions = self.get_other_positions(self.get_state())
        if hunter_positions:
            # For learning, we need to know what the hunter did
            # In practice, this would be observed from the environment
            closest_hunter_pos = min(hunter_positions, 
                                   key=lambda p: self.manhattan_distance(self.pos, p))
            self._previous_hunter_action = closest_hunter_pos
    
    def die(self):
        """Handle prey death by respawning"""
        # Learn from death experience
        if self.last_state is not None and self.last_action is not None:
            death_reward = -10
            hunter_action = getattr(self, '_previous_hunter_action', None)
            self.learn_from_experience(death_reward, hunter_action)
        
        # Reset death flag
        self._is_dead = False
        
        # Respawn at random location
        new_pos = self.get_collision_free_position()
        if new_pos:
            self.model.grid.move_agent(self, new_pos)
            logger.info(f"MinimaxQPrey {self.unique_id} respawned at {new_pos}")
    
    def get_collision_free_position(self):
        """Find a random empty position on the grid"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = self.random.randrange(self.model.grid.width)
            y = self.random.randrange(self.model.grid.height)
            if self.model.grid.is_cell_empty((x, y)):
                return (x, y)
        
        # If no empty cells found, return current position
        logger.warning(f"Could not find empty cell for MinimaxQPrey {self.unique_id}")
        return self.pos
    
    def get_metrics(self):
        """Extend metrics for MinimaxQPrey"""
        metrics = super().get_metrics()
        metrics.update({
            'agent_type': 'MinimaxQPrey',
            'times_caught': max(0, int(-self.total_reward / 10))  # Approximate deaths
        })
        return metrics