import abc
import logging
from mesa import Agent, Model

logger = logging.getLogger(__name__)

class BaseAgent(Agent):
    total_kills = 0

    def __init__(self, model: Model, move_cost=1):
        super().__init__(model)
        self.move_cost = move_cost
        self.kills = 0

    def increment_kills(self):
        self.kills += 1
        type(self).total_kills += 1


    def get_collision_free_position(self):
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

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def pos_to_direction(self, current_pos, action_pos):
        """Convert from current position and action position to direction string."""
        if action_pos is None or current_pos is None:
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

    def get_possible_actions(self, pos=None):
        """
        Get possible actions (neighboring positions) from a given position.
        
        Args:
            pos (tuple, optional): Position to get neighbors for. If None, uses current position.
            
        Returns:
            list: List of neighboring positions (excluding center, non-Moore neighborhood)
        """        
        if pos is None:
            pos = self.pos
        return self.model.grid.get_neighborhood(pos, moore=False, include_center=False)

    def hunt(self, return_reward=False):
        """
        Hunt prey at current position - shared implementation for all hunters
        
        Args:
            return_reward (bool): If True, returns numeric reward (for NashQ agents)
                                 If False, returns boolean success (for other hunters)
        
        Returns:
            int or bool: Reward value if return_reward=True, success boolean otherwise
        """
        # Get all agents at this position
        cell_agents = self.model.grid.get_cell_list_contents([self.pos])
        
        # Initialize result variables
        success = False
        reward = 0
        
        # Look for prey
        for agent in cell_agents:
            if agent.__class__.__name__.endswith("Prey"):                # Only hunt if prey is not already scheduled for removal
                if not getattr(agent, 'scheduled_for_removal', False):                    
                    # Mark the prey as caught/eaten
                    logger.info(f"{self.__class__.__name__} {self.unique_id} ate {agent.__class__.__name__} {agent.unique_id} at {self.pos}")
                    
                    # Register kill for visualization
                    self.model.register_kill(self, agent)
                    
                    # Set a flag to indicate the prey is scheduled for removal
                    if hasattr(agent, '_is_dead'):
                        agent._is_dead = True
                    else:
                        agent.scheduled_for_removal = True
                    
                    # Schedule prey for respawn in next step
                    self.model.pending_prey_respawns.append(agent)
                    
                    # Increment kill counter
                    self.increment_kills()
                    
                    # Schedule hunter teleportation next step
                    self.model.pending_hunter_teleports.append(self)
                    
                    success = True
                    reward = 10  # Reward for successful hunt
                    break
        
        return reward if return_reward else success

    def random_move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        new_pos = self.random.choice(neighbors)
        self.model.grid.move_agent(self, new_pos)
        logger.debug(f"{self.__class__.__name__} {self.unique_id} moved to {new_pos}")

    def step(self):

        self.random_move()

    def get_state(self, agent_type_filter):
        """Get current state: (my_position, filtered_agent_positions)"""
        filtered_positions = tuple(sorted([
            agent.pos for agent in self.model.agents
            if isinstance(agent, agent_type_filter)
        ]))
        return (self.pos, filtered_positions)

    def get_metrics(self):
        """Get performance metrics for this agent."""
        return {
            'agent_id': self.unique_id,
            'agent_type': self.__class__.__name__,
            'position': self.pos
        }

