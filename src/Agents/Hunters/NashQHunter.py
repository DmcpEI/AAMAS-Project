import logging
import numpy as np
from Agents.NashQAgent import NashQAgent

logger = logging.getLogger(__name__)


class NashQHunter(NashQAgent):
    total_kills = 0  # Class variable to track total kills across all instances
    
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.9, move_cost=1):
        super().__init__(model, alpha, gamma, epsilon, move_cost)
        self.Q = {}  # Q-table: (state, hunter_action, prey_action) -> value        self.epsilon_min = 0.1  # Higher minimum exploration for hunters
        self.epsilon_decay = 0.998  # Slower decay to maintain exploration longer

    def get_state(self):
        """Get current state: (my_position, prey_positions)"""
        # Get all prey positions
        prey_positions = tuple(sorted([
            agent.pos for agent in self.model.agents 
            if agent.__class__.__name__.endswith("Prey")
        ]))
        return (self.pos, prey_positions)

    def get_possible_other_actions(self, prey_pos):
        return self.get_possible_actions(prey_pos)    
        
    def get_other_positions(self, state):
        return state[1]
        
    def get_other_agent_q_table(self):
        """Get the Q-table of a NashQPrey for Nash equilibrium computation."""
        for agent in self.model.agents:
            if agent.__class__.__name__ == "NashQPrey":
                return getattr(agent, 'Q', {})
        return {}      
    def step(self):
        """Normal step method - works for both synchronized and non-synchronized modes."""
        if hasattr(self.model, 'nash_q_phase') and self.model.nash_q_phase == "learning":
            # During learning phase, don't move - just learn
            return
        
        # Normal movement behavior
        state = self.get_state()
        action = self.select_action(state)
        old_pos = self.pos
        
        # Move to the selected position
        if action and action != self.pos:
            self.model.grid.move_agent(self, action)
        
        # Hunt is handled by centralized _check_and_perform_hunting() to avoid timing issues
        # hunt_result = self.hunt(return_reward=True)
        
        # For now, just calculate movement cost as reward since hunting is centralized
        reward = -0.1  # Movement cost
        print(f" Hunter {self.unique_id}: Moved {old_pos}→{self.pos}, Reward: {reward:.1f} (hunt handled centrally)")
        
        # Store reward for model's reward system
        self._step_reward = reward

    def observe_state(self):
        """Observe current state at the beginning of the step - PHASE 1."""
        self.observed_state = self.get_state()
        # Store the observed state for later use in learning phase
        return self.observed_state

    def choose_nash_q_action(self):
        """Choose action during Nash Q-Learning Phase 1: Action Selection."""
        if not hasattr(self, 'observed_state'):
            self.observed_state = self.get_state()
        
        # Use Nash Q-Learning action selection
        action = self.select_action(self.observed_state)
        return action