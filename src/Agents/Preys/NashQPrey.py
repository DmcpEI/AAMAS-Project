import numpy as np
from Agents.NashQAgent import NashQAgent
# Removed circular import: from Agents.Hunters.NashQHunter import NashQHunter

class NashQPrey(NashQAgent):
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.3, move_cost=1):
        super().__init__(model, alpha, gamma, epsilon, move_cost)
        self.Q = {}  # Q-table: (state, prey_action, hunter_action) -> value        self.epsilon_min = 0.1  # Higher minimum exploration for prey
        self.epsilon_decay = 0.998  # Slower decay to maintain exploration longer
        
    def get_state(self):
        """Get current state: (my_position, hunter_positions)"""
        # Get all hunter positions       
        hunter_positions = tuple(sorted([
            agent.pos for agent in self.model.agents 
            if agent.__class__.__name__.endswith("Hunter")
        ]))
        return (self.pos, hunter_positions)

    def get_possible_other_actions(self, hunter_pos):
        return self.get_possible_actions(hunter_pos)
        
    def get_other_positions(self, state):
        return state[1]
        
    def get_other_agent_q_table(self):
        """Get the Q-table of a NashQHunter for Nash equilibrium computation."""
        for agent in self.model.agents:
            if agent.__class__.__name__ == "NashQHunter":

                return getattr(agent, 'Q', {})
        return {}
    
    def step(self):
        """Step method - only called for non-Nash Q phases or fallback."""
        # This should not be called when using the synchronized Nash Q system
        # But kept for compatibility with other agent types
        if hasattr(self.model, 'nash_q_phase') and self.model.nash_q_phase == "execution":
            # During execution phase, agents are moved by the model
            return        
        
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

    def die(self):
        # Apply penalty for being caught before teleporting
        reward = -10
        if self.last_state is not None and self.last_action is not None:
            # For death, there's no next state, so we just update with the penalty
            # Using a simple Q-learning update for terminal state
            key_other_action = self.last_other_action if self.last_other_action is not None else "unknown"
            old_q = self.Q.get((self.last_state, self.last_action, key_other_action), 0.0)
            # No next state, so future reward is 0 (terminal state)
            new_q = old_q + self.alpha * (reward - old_q)
            self.Q[(self.last_state, self.last_action, key_other_action)] = new_q

              # Debug print with readable format
            state_str = f"[pos={self.last_state[0]}, hunters={list(self.last_state[1])}]"
            action_dir = self.pos_to_direction(self.last_state[0], self.last_action)
            hunter_action_dir = self.pos_to_direction(self.last_state[1][0] if self.last_state[1] else None, self.last_other_action) if self.last_other_action else "None"
            print(f"Prey {self.unique_id} Nash Q-update on death: state={state_str}, action={action_dir}, hunter_action={hunter_action_dir} -> Q={new_q:.3f} (reward={reward})")        # Track penalty for visualization
        self._step_reward = reward
        # Teleport to a collision-free random empty cell
        new_pos = self.get_collision_free_position()
        if new_pos:

            self.model.grid.move_agent(self, new_pos)