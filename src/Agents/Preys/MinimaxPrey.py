"""
Minimax Prey Agent
=================

Prey agent that uses minimax algorithm with alpha-beta pruning to maximize
survival time by evading hunters. The prey is the minimizing player in the
minimax tree search.
"""

import logging
from typing import Tuple, List
from Agents.MinimaxAgent import MinimaxAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class MinimaxPrey(MinimaxAgent):
    """
    Prey agent using minimax algorithm.
    Minimizes capture probability while maximizing distance from hunters.
    """
    def __init__(self, model, search_depth=3):
        super().__init__(model, search_depth)
        self.move_cost = 1  # Cost for each move
        self._is_dead = False  # Flag to prevent movement after death
        self.exploration_rate = 0.15  # 15% chance of random exploration to break patterns
        
    def get_state(self):
        filtered_positions = tuple(sorted([
            agent.pos for agent in self.model.agents
            if hasattr(agent, 'step') and agent.__class__.__name__.endswith("Hunter")
        ]))
        return (self.pos, filtered_positions)
        
    def evaluate_state(self, state):
        """
        Evaluate state from prey's perspective (minimizing player).
        Lower values are better for the prey (since it's minimizing).
        
        Returns:
            float: State evaluation score
        """
        my_pos, hunter_positions = state
        
        # If no hunters, return best possible score for prey
        if not hunter_positions:
            return -1000  # Very low score (good for minimizing player)
            
        # Check if prey was caught (losing condition)
        if my_pos in hunter_positions:
            return 1000  # Very high score (bad for minimizing player)
            
        # Strategic evaluation for prey survival
        total_danger = 0
        
        for hunter_pos in hunter_positions:
            distance = self.manhattan_distance(my_pos, hunter_pos)
            
            # Immediate danger based on distance
            if distance == 0:
                return 1000  # Caught!
            elif distance == 1:
                total_danger += 100  # Very close
            elif distance == 2:
                total_danger += 30   # Close
            else:
                total_danger += max(0, 10 - distance)  # Decreasing danger
            
            # Predict hunter movement - penalize positions hunters might move to
            hunter_next_moves = self.model.grid.get_neighborhood(hunter_pos, moore=False, include_center=False)
            if my_pos in hunter_next_moves:
                total_danger += 50  # Hunter can reach us next turn
                
        # Strategic positioning
        # Prefer positions with more escape routes
        escape_routes = len(self.model.grid.get_neighborhood(my_pos, moore=False, include_center=False))
        mobility_bonus = -escape_routes * 5  # More mobility is better (negative score)
        
        # Prefer center positions over corners when far from hunters
        center_x, center_y = self.model.grid.width // 2, self.model.grid.height // 2
        center_distance = self.manhattan_distance(my_pos, (center_x, center_y))
        
        # Only prefer center if not too close to hunters
        min_hunter_distance = min(self.manhattan_distance(my_pos, h_pos) for h_pos in hunter_positions)
        if min_hunter_distance > 3:
            center_bonus = -center_distance * 2  # Slight preference for center when safe
        else:
            center_bonus = center_distance * 1  # Prefer edges when in danger
            
        # Add randomization to break ties and prevent loops
        randomization = self.random.uniform(-2, 2)
        
        total_score = total_danger + mobility_bonus + center_bonus + randomization
        
        logger.debug(f"Prey {self.unique_id} strategic evaluation: {total_score:.2f}")
        
        return total_score
    
    def _is_winning_move(self, state):
        """Prey doesn't have explicit winning moves, only survival"""
        return False
    
    def _is_losing_move(self, state):
        """Check if this state represents a prey loss (being caught)"""
        my_pos, hunter_positions = state
        return my_pos in hunter_positions
    
    def select_action(self, state):
        """Select best action using minimax algorithm (minimizing)"""
        self.nodes_searched = 0  # Reset counter
        
        # For prey, we start as minimizing player
        eval_score, best_action = self.minimax(state, self.search_depth, False)
        
        # Debug output        logger.debug(f"MinimaxPrey {self.unique_id}: Minimax search completed")
        logger.debug(f"  Nodes searched: {self.nodes_searched}")
        logger.debug(f"  Best action: {best_action}")
        logger.debug(f"  Evaluation: {eval_score:.3f}")
        return best_action if best_action else self._get_fallback_action()

    def get_best_action(self):
        """Get the best action for the agent"""
        state = self.get_state()
        # For prey, we start as minimizing player
        _, best_action = self.minimax(state, self.search_depth, False)
        return best_action

    def step(self):
        """Single step execution: evaluate and move."""
        if self._is_dead:
            logger.debug(f"MinimaxPrey {self.unique_id} skipping step (dead)")
            return
            
        logger.debug(f"MinimaxPrey {self.unique_id} step started at {self.pos}")
        
        # Get current state
        current_state = self.get_state()
        
        # Select best action using minimax
        best_action = self.select_action(current_state)
        
        # Move to the selected position
        if best_action and best_action != self.pos:
            self.model.grid.move_agent(self, best_action)
            logger.info(f"MinimaxPrey {self.unique_id} moved from {current_state[0]} to {best_action}")
        else:
            logger.debug(f"MinimaxPrey {self.unique_id} stayed at {self.pos}")
    
    def die(self):
        """Handle prey death by scheduling respawn for next step."""
        # Mark as dead to prevent further movement this step        self._is_dead = True
        # Add to pending respawns instead of immediately respawning
        self.model.pending_prey_respawns.append(self)
        logger.info(f"MinimaxPrey {self.unique_id} scheduled for respawn next step")
    
    def get_metrics(self):
        """Extend performance metrics for MinimaxPrey."""
        metrics = super().get_metrics()
        metrics.update({
            'search_depth': self.search_depth,
            'is_dead': self._is_dead
        })
        return metrics
