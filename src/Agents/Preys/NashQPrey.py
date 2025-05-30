# filepath: c:\Users\guilh\OneDrive - Universidade da Madeira\mestrado\AASMA\AAMAS-Project\src\Agents\Preys\NashQPrey.py
import numpy as np
from Agents.NashQAgent import NashQAgent

def pos_to_direction(current_pos, action_pos):
    """Convert from current position and action position to direction string."""
    if action_pos is None:
        return "None"
    
    dx = action_pos[0] - current_pos[0]
    dy = action_pos[1] - current_pos[1]
    
    if dx == 0 and dy == 1:
        return "Up"
    elif dx == 0 and dy == -1:
        return "Down"
    elif dx == 1 and dy == 0:
        return "Right"
    elif dx == -1 and dy == 0:
        return "Left"
    else:
        return f"({dx},{dy})"  # Fallback for unexpected moves

class NashQPrey(NashQAgent):
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.1, move_cost=1):
        super().__init__(model, alpha, gamma, epsilon, move_cost)
        self.Q = {}  # Q-table: (state, prey_action, hunter_action) -> value

    def get_state(self):
        # State: own position and sorted tuple of hunter positions
        hunter_positions = tuple(sorted([
            agent.pos for agent in self.model.agents
            if hasattr(agent, 'step') and agent.__class__.__name__.endswith("Hunter")        ]))
        return (self.pos, hunter_positions)

    def get_possible_actions(self):
        return self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)

    def get_possible_other_actions(self, hunter_pos):
        return self.model.grid.get_neighborhood(hunter_pos, moore=False, include_center=False)

    def get_other_positions(self, state):
        return state[1]
    
    def get_other_agent_q_table(self):
        """Get the Q-table of a NashQHunter for Nash equilibrium computation."""
        for agent in self.model.agents:
            if agent.__class__.__name__ == "NashQHunter":
                return getattr(agent, 'Q', None)
        return None

    def step(self):
        # Debug: Print current state
        print(f"Prey {self.unique_id} step: last_state={self.last_state is not None}, last_action={self.last_action is not None}, last_other_action={self.last_other_action is not None}")
        
        # --- Nash Q-table update for previous transition ---
        if self.last_state is not None and self.last_action is not None:
            current_state = self.get_state()
            new_q = self.update_q_nash(self.last_state, self.last_action, self.last_reward, current_state, self.last_other_action)
            
            # Debug print with readable format
            state_str = f"[pos={self.last_state[0]}, hunters={list(self.last_state[1])}]"
            action_dir = pos_to_direction(self.last_state[0], self.last_action)
            hunter_action_dir = pos_to_direction(self.last_state[1][0] if self.last_state[1] else None, self.last_other_action) if self.last_other_action else "None"
            print(f"Prey {self.unique_id} Nash Q-update: state={state_str}, action={action_dir}, hunter_action={hunter_action_dir} -> Q={new_q:.3f} (reward={self.last_reward})")
        # --- End Nash Q-table update ---
        
        state = self.get_state()
        # For simplicity, assume only one hunter and get its last action if available
        hunter_action = None
        hunter_positions = state[1]
        
        # Debug: Print state information with readable format
        state_str = f"[pos={state[0]}, hunters={list(state[1])}]"
        print(f"Prey {self.unique_id} current state: {state_str}")
        if hunter_positions:
            for agent in self.model.agents:
                if hasattr(agent, 'step') and agent.__class__.__name__.endswith("Hunter"):
                    if hasattr(agent, 'last_action'):
                        hunter_action = agent.last_action
                        hunter_action_dir = pos_to_direction(hunter_positions[0] if hunter_positions else None, hunter_action)
                        print(f"Prey {self.unique_id} found hunter action: {hunter_action_dir}")
                    break
        
        action = self.select_action(state, hunter_action)
        self.model.grid.move_agent(self, action)
        # Reward: -0.1 for moving, +5 for surviving a step, -10 if removed (caught)
        reward = -0.1
        reward += 5  # survived this step
        
        # Track reward for visualization
        self._step_reward = reward
        
        # Store for next Q-update
        self.last_state = state
        self.last_action = action
        self.last_other_action = hunter_action
        self.last_reward = reward
          # Debug: Print what we're storing with readable format
        store_state_str = f"[pos={state[0]}, hunters={list(state[1])}]"
        action_dir = pos_to_direction(state[0], action)
        hunter_action_dir = pos_to_direction(hunter_positions[0] if hunter_positions else None, hunter_action) if hunter_action else "None"
        print(f"Prey {self.unique_id} storing: state={store_state_str}, action={action_dir}, hunter_action={hunter_action_dir}, reward={reward}")

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
            action_dir = pos_to_direction(self.last_state[0], self.last_action)
            hunter_action_dir = pos_to_direction(self.last_state[1][0] if self.last_state[1] else None, self.last_other_action) if self.last_other_action else "None"
            print(f"Prey {self.unique_id} Nash Q-update on death: state={state_str}, action={action_dir}, hunter_action={hunter_action_dir} -> Q={new_q:.3f} (reward={reward})")
        # Track penalty for visualization
        self._step_reward = reward
        # Teleport to a random empty cell
        empty_cells = [cell for cell in self.model.grid.empties]
        if empty_cells:
            new_pos = self.random.choice(empty_cells)
            self.model.grid.move_agent(self, new_pos)
        # Optionally, reset any other state if needed (not Q-table)
        # Do NOT remove the agent from the model
