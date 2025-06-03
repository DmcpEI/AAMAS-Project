import logging
import numpy as np
from Agents.NashQAgent import NashQAgent
# Removed circular import: from Agents.Preys.NashQPrey import NashQPrey

logger = logging.getLogger(__name__)


class NashQHunter(NashQAgent):
    total_kills = 0  # Class variable to track total kills across all instances
    
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.1, move_cost=1):
        super().__init__(model, alpha, gamma, epsilon, move_cost)
        self.Q = {}  # Q-table: (state, hunter_action, prey_action) -> value

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
        """Main step method that handles movement, hunting, and Q-learning."""

        # --- Nash Q-table update for previous transition ---
        if self.last_state is not None and self.last_action is not None:
            current_state = self.get_state()
            new_q = self.update_q_nash(self.last_state, self.last_action, self.last_reward, current_state, self.last_other_action)

              # Debug print with readable format
            state_str = f"[pos={self.last_state[0]}, preys={list(self.last_state[1])}]"
            action_dir = self.pos_to_direction(self.last_state[0], self.last_action)
            prey_action_dir = self.pos_to_direction(self.last_state[1][0] if self.last_state[1] else None, self.last_other_action) if self.last_other_action else "None"

            print(f"Hunter {self.unique_id} Nash Q-update: state={state_str}, action={action_dir}, prey_action={prey_action_dir} -> Q={new_q:.3f} (reward={self.last_reward})")
        # --- End Nash Q-table update ---

        state = self.get_state()
        prey_action = None
        prey_positions = state[1]
        
        # Debug: Print state information with readable format
        state_str = f"[pos={state[0]}, preys={list(state[1])}]"
        print(f"Hunter {self.unique_id} current state: {state_str}")

        
        # Get prey action for Nash Q-learning

        if prey_positions:
            for agent in self.model.agents:
                if hasattr(agent, 'step') and agent.__class__.__name__.endswith("Prey"):
                    if hasattr(agent, 'last_action'):
                        prey_action = agent.last_action

                        prey_action_dir = self.pos_to_direction(prey_positions[0] if prey_positions else None, prey_action)
                        print(f"Hunter {self.unique_id} found prey action: {prey_action_dir}")
                    break
        
        # Select action and move
        action = self.select_action(state, prey_action)
        self.model.grid.move_agent(self, action)
          # Hunt after moving using shared method
        hunt_reward = self.hunt(return_reward=True)
        
        # Calculate reward
        if hunt_reward > 0:
            # If caught prey, only get catch reward (no step penalty)
            reward = hunt_reward
        else:
            # If no catch, get step penalty (matches prey's +1 survival reward)
            reward = -1

        
        # Track reward for visualization
        self._step_reward = reward
        

        # Store state and action for next update
        self.last_state = state
        self.last_action = action
        self.last_other_action = prey_action
        self.last_reward = reward        # Debug: Print what we're storing with readable format
        store_state_str = f"[pos={state[0]}, preys={list(state[1])}]"
        action_dir = self.pos_to_direction(state[0], action)
        prey_action_dir = self.pos_to_direction(prey_positions[0] if prey_positions else None, prey_action) if prey_action else "None"
        print(f"Hunter {self.unique_id} storing: state={store_state_str}, action={action_dir}, prey_action={prey_action_dir}, reward={reward}")

