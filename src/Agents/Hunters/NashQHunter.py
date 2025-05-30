# filepath: c:\Users\guilh\OneDrive - Universidade da Madeira\mestrado\AASMA\AAMAS-Project\src\Agents\Hunters\NashQHunter.py
import logging
import numpy as np
from Agents.NashQAgent import NashQAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

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

class NashQHunter(NashQAgent):
    total_kills = 0  # Class variable to track total kills across all instances
    
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.1, move_cost=1):
        super().__init__(model, alpha, gamma, epsilon, move_cost)
        self.Q = {}  # Q-table: (state, hunter_action, prey_action) -> value

    def get_state(self):
        # Get all prey positions (both regular Prey and NashQPrey)
        prey_positions = tuple(sorted([
            agent.pos for agent in self.model.agents 
            if isinstance(agent, Prey) or agent.__class__.__name__ == "NashQPrey"        ]))
        return (self.pos, prey_positions)

    def get_possible_actions(self):
        return self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)

    def get_possible_other_actions(self, prey_pos):
        return self.model.grid.get_neighborhood(prey_pos, moore=False, include_center=False)

    def get_other_positions(self, state):
        return state[1]
    
    def get_other_agent_q_table(self):
        """Get the Q-table of a NashQPrey for Nash equilibrium computation."""
        for agent in self.model.agents:
            if agent.__class__.__name__ == "NashQPrey":
                return getattr(agent, 'Q', None)
        return None

    def step(self):
        # Debug: Print current state
        print(f"Hunter {self.unique_id} step: last_state={self.last_state is not None}, last_action={self.last_action is not None}, last_other_action={self.last_other_action is not None}")
        
        # --- Nash Q-table update for previous transition ---
        if self.last_state is not None and self.last_action is not None:
            current_state = self.get_state()
            new_q = self.update_q_nash(self.last_state, self.last_action, self.last_reward, current_state, self.last_other_action)
            
            # Debug print with readable format
            state_str = f"[pos={self.last_state[0]}, preys={list(self.last_state[1])}]"
            action_dir = pos_to_direction(self.last_state[0], self.last_action)
            prey_action_dir = pos_to_direction(self.last_state[1][0] if self.last_state[1] else None, self.last_other_action) if self.last_other_action else "None"
            print(f"Hunter {self.unique_id} Nash Q-update: state={state_str}, action={action_dir}, prey_action={prey_action_dir} -> Q={new_q:.3f} (reward={self.last_reward})")
        # --- End Nash Q-table update ---

        state = self.get_state()
        prey_action = None
        prey_positions = state[1]
        
        # Debug: Print state information with readable format
        state_str = f"[pos={state[0]}, preys={list(state[1])}]"
        print(f"Hunter {self.unique_id} current state: {state_str}")
        if prey_positions:
            for agent in self.model.agents:
                if hasattr(agent, 'step') and agent.__class__.__name__.endswith("Prey"):
                    if hasattr(agent, 'last_action'):
                        prey_action = agent.last_action
                        prey_action_dir = pos_to_direction(prey_positions[0] if prey_positions else None, prey_action)
                        print(f"Hunter {self.unique_id} found prey action: {prey_action_dir}")
                    break
        
        action = self.select_action(state, prey_action)
        self.model.grid.move_agent(self, action)
        reward = -0.1
        reward += self.hunt()
        
        # Track reward for visualization
        self._step_reward = reward
        
        # Store for next Q-update
        self.last_state = state
        self.last_action = action
        self.last_other_action = prey_action
        self.last_reward = reward
        
        # Debug: Print what we're storing with readable format
        store_state_str = f"[pos={state[0]}, preys={list(state[1])}]"
        action_dir = pos_to_direction(state[0], action)
        prey_action_dir = pos_to_direction(prey_positions[0] if prey_positions else None, prey_action) if prey_action else "None"
        print(f"Hunter {self.unique_id} storing: state={store_state_str}, action={action_dir}, prey_action={prey_action_dir}, reward={reward}")

    def hunt(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        reward = 0
        for other in list(cellmates):            
            if isinstance(other, Prey) or other.__class__.__name__ == "NashQPrey":
                logger.info(f"NashQHunter {self.unique_id} ate Prey {other.unique_id} at {self.pos}")
                if hasattr(other, 'die'):
                    other.die()
                self.increment_kills()
                NashQHunter.total_kills += 1
                reward += 10  # reward for eating prey
                # Teleport hunter to a random empty cell after a successful catch
                empty_cells = [cell for cell in self.model.grid.empties]
                if empty_cells:
                    new_pos = self.random.choice(empty_cells)
                    self.model.grid.move_agent(self, new_pos)
        return reward
