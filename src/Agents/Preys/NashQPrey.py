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

    def move_phase(self):
        """Phase 1: Movement only."""
        # Debug: Print current state
        print(f"Prey {self.unique_id} move_phase: last_state={self.last_state is not None}, last_action={self.last_action is not None}, last_other_action={self.last_other_action is not None}")
        
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
        
        # Store state and action for next update
        self.last_state = state
        self.last_action = action
        self.last_other_action = hunter_action
        
        # Debug: Print what we're storing with readable format
        store_state_str = f"[pos={state[0]}, hunters={list(state[1])}]"
        action_dir = pos_to_direction(state[0], action)
        hunter_action_dir = pos_to_direction(hunter_positions[0] if hunter_positions else None, hunter_action) if hunter_action else "None"
        print(f"Prey {self.unique_id} storing: state={store_state_str}, action={action_dir}, hunter_action={hunter_action_dir}")

    def action_phase(self):
        """Phase 2: Survival reward calculation."""
        # Zero-sum reward: +1 for surviving a step (matches hunter's -1 step penalty)
        reward = 1  # survived this step
        
        # Track reward for visualization
        self._step_reward = reward
        
        # Update stored reward for next Q-update
        self.last_reward = reward
        
        print(f"Prey {self.unique_id} action_phase: reward={reward}")

    def step(self):
        """Legacy step method - calls both phases for backward compatibility."""
        self.move_phase()
        self.action_phase()

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
            print(f"Prey {self.unique_id} Nash Q-update on death: state={state_str}, action={action_dir}, hunter_action={hunter_action_dir} -> Q={new_q:.3f} (reward={reward})")        # Track penalty for visualization
        self._step_reward = reward
        # Teleport to a collision-free random empty cell
        new_pos = self._get_collision_free_position()
        if new_pos:
            self.model.grid.move_agent(self, new_pos)
        # Optionally, reset any other state if needed (not Q-table)
        # Do NOT remove the agent from the model

    def _get_collision_free_position(self):
        """Get a random empty position that doesn't have other agents."""
        empty_cells = [cell for cell in self.model.grid.empties]
        if not empty_cells:
            return None
        
        # Filter out cells that already have agents
        available_cells = []
        for cell in empty_cells:
            cell_contents = self.model.grid.get_cell_list_contents([cell])
            # Only consider truly empty cells (no agents)
            if not cell_contents:
                available_cells.append(cell)
        
        if available_cells:
            return self.random.choice(available_cells)
        else:
            # Fallback to any empty cell if no completely empty cells available
            return self.random.choice(empty_cells) if empty_cells else None
