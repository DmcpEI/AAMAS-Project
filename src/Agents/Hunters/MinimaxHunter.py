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
    def __init__(self, model, search_depth=3):  # Reduced back to 3 for performance
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
            return 0        # Check if hunter caught any prey (winning condition)
        if my_pos in prey_positions:
            return 1000  # Very high reward for catching prey

        # Calculate strategic evaluation
        scores = []
        for prey_pos in prey_positions:
            distance = self.manhattan_distance(my_pos, prey_pos)

            # Base distance score (closer is better) - MUCH stronger weight on distance
            distance_score = -distance * 10 # Increased from 10 to 25
            
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
        """Single step execution: move and hunt."""
        logger.debug(f"MinimaxHunter {self.unique_id} step started at {self.pos}")
        

        # Get current state
        current_state = self.get_state()
        my_pos, prey_positions = current_state
        
        best_action = self.select_action(current_state)
        

        # Move to the selected position
        if best_action and best_action != self.pos:
            self.model.grid.move_agent(self, best_action)
            logger.info(f"MinimaxHunter {self.unique_id} moved from {current_state[0]} to {best_action}")
        else:
            logger.debug(f"MinimaxHunter {self.unique_id} stayed at {self.pos}")
            self.stuck_counter += 1  # Count staying in place as being stuck

        # Hunt at current position
        self.hunt()
    def get_metrics(self):
        """Extend performance metrics for MinimaxHunter."""
        metrics = super().get_metrics()
        metrics.update({
            'search_depth': self.search_depth,
            'last_nodes_searched': getattr(self, 'nodes_searched', 0),
            'total_kills': MinimaxHunter.total_kills
        })
        return metrics
