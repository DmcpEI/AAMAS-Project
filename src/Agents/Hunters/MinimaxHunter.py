"""
Minimax Hunter Agent
===================

Hunter agent that uses minimax algorithm with alpha-beta pruning to maximize
the probability of catching prey. The hunter is the maximizing player in the
minimax tree search.
"""

import logging
from typing import Tuple, List
from Agents.MinimaxAgent import MinimaxAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class MinimaxHunter(MinimaxAgent):
    """
    Hunter agent using minimax algorithm.
    Maximizes capture probability while minimizing distance to prey.
    """
    
    total_kills = 0  # Class variable to track total kills across all instances
    
    def __init__(self, model, search_depth=3):
        super().__init__(model, search_depth)
        self.move_cost = 1  # Cost for each move
        
    def get_state(self):
        """Get current state: (my_position, prey_positions)"""
        # Get all prey positions (both regular Prey and other prey types)
        prey_positions = tuple(sorted([
            agent.pos for agent in self.model.agents 
            if isinstance(agent, Prey) or agent.__class__.__name__.endswith("Prey")
        ]))
        return (self.pos, prey_positions)
    def evaluate_state(self, state):
        """
        Evaluate state from hunter's perspective (maximizing player).
        Higher values are better for the hunter.
        
        Returns:
            float: State evaluation score
        """
        my_pos, prey_positions = state
        
        # If no prey, return neutral score
        if not prey_positions:
            return 0
            
        # Check if hunter caught any prey (winning condition)
        if my_pos in prey_positions:
            return 1000  # Very high reward for catching prey
            
        # Calculate strategic evaluation
        scores = []
        
        for prey_pos in prey_positions:
            distance = self.manhattan_distance(my_pos, prey_pos)
            
            # Base distance score (closer is better)
            distance_score = -distance * 10
            
            # Strategic positioning: prefer positions that limit prey escape routes
            escape_routes = len(self.model.grid.get_neighborhood(prey_pos, moore=False, include_center=False))
            blocking_score = (4 - escape_routes) * 5  # Reward limiting escape routes
            
            # Prefer cornering prey (prey closer to edges)
            corner_score = 0
            if prey_pos[0] == 0 or prey_pos[0] == self.model.grid.width - 1:
                corner_score += 3
            if prey_pos[1] == 0 or prey_pos[1] == self.model.grid.height - 1:
                corner_score += 3
                
            # Anticipation: reward positions that can intercept likely prey moves  
            anticipation_score = 0
            for next_prey_pos in self.model.grid.get_neighborhood(prey_pos, moore=False, include_center=False):
                if self.manhattan_distance(my_pos, next_prey_pos) <= 2:
                    anticipation_score += 2
            
            total_prey_score = distance_score + blocking_score + corner_score + anticipation_score
            scores.append(total_prey_score)
        
        # Take the best opportunity (highest scoring prey)
        best_score = max(scores) if scores else 0
        
        # Add slight randomization to break ties
        randomization = self.random.uniform(-1, 1)
        
        total_score = best_score + randomization
        
        logger.debug(f"Hunter {self.unique_id} strategic evaluation: {total_score:.2f}")
        
        return total_score
    
    def _is_winning_move(self, state):
        """Check if this state represents a hunter win (catching prey)"""
        my_pos, prey_positions = state
        return my_pos in prey_positions
    
    def _is_losing_move(self, state):
        """Hunter doesn't have explicit losing moves, only suboptimal ones"""
        return False
    
    def step(self):
        """Execute one step of the hunter agent."""
        logger.debug(f"MinimaxHunter {self.unique_id} step started at {self.pos}")
        
        # Get current state
        current_state = self.get_state()
        
        # Select best action using minimax
        best_action = self.select_action(current_state)
        
        # Move to the selected position
        if best_action and best_action != self.pos:
            self.model.grid.move_agent(self, best_action)
            logger.info(f"MinimaxHunter {self.unique_id} moved from {current_state[0]} to {best_action}")
        else:
            logger.debug(f"MinimaxHunter {self.unique_id} stayed at {self.pos}")
          # Hunt for prey at current position
        self.hunt()
    
    def hunt(self):
        """Hunt for prey at current position."""
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for other in list(cellmates):
            if isinstance(other, Prey) or other.__class__.__name__.endswith("Prey"):
                logger.info(f"MinimaxHunter {self.unique_id} caught prey {other.unique_id} at {self.pos}")
                self.model.register_kill(self, other)
                if hasattr(other, 'die'):
                    other.die()
                self.increment_kills()
                # Schedule teleportation next step
                self.model.pending_hunter_teleports.append(self)
                return True
        return False
    
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
    
    def get_metrics(self):
        """Get performance metrics for this hunter."""
        return {
            'agent_id': self.unique_id,
            'agent_type': 'MinimaxHunter',
            'position': self.pos,
            'search_depth': self.search_depth,
            'last_nodes_searched': getattr(self, 'nodes_searched', 0),
            'total_kills': MinimaxHunter.total_kills
        }
