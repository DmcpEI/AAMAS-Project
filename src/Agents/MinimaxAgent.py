"""
Minimax Agent     def __init__(self, model, search_depth=2):
        super().__init__(model)
        self.search_depth = search_depth
        self.nodes_searched = 0  # For performance tracking
        self.exploration_rate = 0.0  # 0% chance of random exploration (deterministic behavior)
        self.previous_moves = []  # Track recent moves to avoid loops
        self.max_history = 5  # Remember last 5 movesass
=======================

Base class for minimax-based agents in the hunter-prey simulation.
Implements the minimax algorithm with alpha-beta pruning for competitive gameplay.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any
from Agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)

class MinimaxAgent(BaseAgent, ABC):
    """
    Base class for minimax-based agents.
    Implements minimax algorithm with alpha-beta pruning.
    """    
    def __init__(self, model, search_depth=2):
        super().__init__(model)
        self.search_depth = search_depth
        self.nodes_searched = 0  # For performance tracking
        self.exploration_rate = 0.0  # 0% chance of random exploration
        self.previous_moves = []  # Track recent moves to avoid loops
        self.max_history = 5  # Remember last 5 moves
        
    def get_possible_actions(self, pos=None):
        """Get possible actions from a given position (or current position)"""
        if pos is None:
            pos = self.pos
        return self.model.grid.get_neighborhood(pos, moore=False, include_center=False)
        
    def is_terminal_state(self, state):
        """Check if state is terminal (game over)"""
        my_pos, other_positions = state
        
        # Terminal if my position overlaps with any other agent
        return my_pos in other_positions
        
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def minimax(self, state, depth, is_maximizing_player, alpha=float('-inf'), beta=float('inf')):
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            state: Current game state (my_pos, other_positions)
            depth: Remaining search depth
            is_maximizing_player: True if maximizing player's turn
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Tuple[float, Optional[Tuple]]: (evaluation_score, best_action)
        """
        self.nodes_searched += 1
        
        # Base case: terminal state or max depth reached
        if depth == 0 or self.is_terminal_state(state):
            return self.evaluate_state(state), None
            
        my_pos, other_positions = state
        
        if is_maximizing_player:
            return self._maximize(state, depth, alpha, beta)
        else:
            return self._minimize(state, depth, alpha, beta)
    
    def _maximize(self, state, depth, alpha, beta):
        """Maximizing player's turn"""
        my_pos, other_positions = state
        max_eval = float('-inf')
        best_action = None
        
        # Try all possible actions for maximizing player
        for action in self.get_possible_actions(my_pos):
            new_state = self._apply_my_action(state, action)
            
            # Quick win check for hunters
            if self._is_winning_move(new_state):
                return 1000, action
                
            eval_score, _ = self.minimax(new_state, depth-1, False, alpha, beta)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Alpha-beta pruning
                
        return max_eval, best_action
    
    def _minimize(self, state, depth, alpha, beta):
        """Minimizing player's turn"""
        my_pos, other_positions = state
        min_eval = float('inf')
        best_action = None
        
        # Try all possible actions for minimizing player
        for action in self.get_possible_actions(my_pos):
            new_state = self._apply_my_action(state, action)
            
            # Avoid losing moves
            if self._is_losing_move(new_state):
                continue
                
            eval_score, _ = self.minimax(new_state, depth-1, True, alpha, beta)
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha-beta pruning
                
        return min_eval, best_action
    
    def _apply_my_action(self, state, action):
        """Apply an action to the current state and return new state"""
        my_pos, other_positions = state
        new_my_pos = action
        return (new_my_pos, other_positions)
    
    def _is_winning_move(self, state):
        """Check if this state represents a win. Override in subclasses."""
        return False
    
    def _is_losing_move(self, state):
        """Check if this state represents a loss. Override in subclasses."""
        return False
    
    def select_action(self, state):
        """Select best action using minimax algorithm"""
        self.nodes_searched = 0  # Reset counter
        
        eval_score, best_action = self.minimax(state, self.search_depth, True)
        
        # Debug output
        agent_type = self.__class__.__name__
        logger.debug(f"{agent_type} {self.unique_id}: Minimax search completed")
        logger.debug(f"  Nodes searched: {self.nodes_searched}")
        logger.debug(f"  Best action: {best_action}")
        logger.debug(f"  Evaluation: {eval_score:.3f}")
        
        return best_action if best_action else self._get_fallback_action()
    
    def _get_fallback_action(self):
        """Fallback action if minimax fails"""
        possible_actions = self.get_possible_actions()
        return self.random.choice(possible_actions) if possible_actions else self.pos
    
    def step(self):
        """Enhanced step with exploration and loop detection"""
        # Random exploration to avoid getting stuck
        if self.random.random() < self.exploration_rate:
            possible_actions = self.get_possible_actions()
            if possible_actions:
                action = self.random.choice(possible_actions)
                self._move_to_action(action)
                logger.debug(f"{self.__class__.__name__} {self.unique_id}: Exploring - moved to {action}")
                return
        
        # Track moves to detect loops
        current_pos = self.pos
        if len(self.previous_moves) >= 2:
            # Check if we're oscillating between two positions
            recent_positions = [move[1] for move in self.previous_moves[-2:]]
            if current_pos in recent_positions and len(set(recent_positions + [current_pos])) <= 2:
                # We're in a loop, try a different action
                logger.debug(f"{self.__class__.__name__} {self.unique_id}: Loop detected, exploring...")
                possible_actions = self.get_possible_actions()
                # Filter out recent positions
                new_actions = [a for a in possible_actions if a not in recent_positions]
                if new_actions:
                    action = self.random.choice(new_actions)
                    self._move_to_action(action)
                    return
        
        # Normal minimax decision
        best_action = self.get_best_action()
        if best_action and best_action != self.pos:
            self._move_to_action(best_action)
    
    def _move_to_action(self, new_pos):
        """Move to new position and track the move"""
        old_pos = self.pos
        self.model.grid.move_agent(self, new_pos)
        
        # Track move history
        self.previous_moves.append((old_pos, new_pos))
        if len(self.previous_moves) > self.max_history:
            self.previous_moves.pop(0)
        
        logger.info(f"{self.__class__.__name__} {self.unique_id} moved from {old_pos} to {new_pos}")
    
    def get_best_action(self):
        """Get the best action for the agent"""
        state = (self.pos, [a.pos for a in self.model.agents if a != self])
        _, best_action = self.minimax(state, self.search_depth, True)
        return best_action
    
    @abstractmethod
    def evaluate_state(self, state):
        """
        Evaluate how good a state is from this agent's perspective.
        Must be implemented by subclasses with different objectives.
        """
        raise NotImplementedError("Subclasses must implement evaluate_state()")
