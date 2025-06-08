"""
Minimax Q-Learning Agent
=======================

Base class for agents using Minimax Q-Learning algorithm.
Combines Q-learning with minimax principles for multi-agent environments.

Unlike regular Q-learning, Minimax Q-learning assumes:
- The other agents act optimally against this agent
- Q-values are updated using minimax backup instead of simple max/min
- Actions are selected based on learned Q-values with exploration
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from Agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)

class MinimaxQAgent(BaseAgent, ABC):
    """
    Base class for Minimax Q-Learning agents.
    
    Key differences from regular Q-Learning:
    - Uses minimax backup for Q-value updates
    - Assumes adversarial environment
    - Q-table stores Q(state, my_action, other_action) values
    """
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.3):
        super().__init__(model)
        
        # Q-Learning parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration
        self.epsilon_decay = 0.995  # Exploration decay
        
        # Q-table: (state, my_action, other_action) -> Q-value
        # This represents Q(s, a_i, a_-i) where a_i is my action and a_-i is other agent's action
        self.Q = {}
        
        # Value function: V(s) = minimax value of state s
        self.V = {}
        
        # Policy: π(s) = probability distribution over actions in state s
        self.policy = {}
        
        # Learning tracking
        self.last_state = None
        self.last_action = None
        self.last_other_action = None
        self._step_reward = 0
        
        # Performance tracking
        self.total_reward = 0
        self.steps_taken = 0
        
    def get_q_value(self, state, my_action, other_action):
        """Get Q-value for state-action pair"""
        key = (state, my_action, other_action)
        return self.Q.get(key, 0.0)
    
    def set_q_value(self, state, my_action, other_action, value):
        """Set Q-value for state-action pair"""
        key = (state, my_action, other_action)
        self.Q[key] = value
    
    def get_all_q_values_for_state(self, state):
        """Get all Q-values for a given state"""
        q_values = {}
        for key, value in self.Q.items():
            if key[0] == state:
                my_action, other_action = key[1], key[2]
                if my_action not in q_values:
                    q_values[my_action] = {}
                q_values[my_action][other_action] = value
        return q_values
    def select_action(self, state):
        """
        Select action using the current policy computed from minimax Q-values.
        
        In Minimax-Q, we maintain a mixed strategy (policy) for each state
        that represents the minimax solution.
        """
        possible_actions = self.get_possible_actions()
        if not possible_actions:
            # No valid actions available - this should not happen in a well-designed environment
            # For Minimax agents, we should never allow staying in place when include_center=False
            raise RuntimeError(f"No valid actions available for {self.__class__.__name__} at position {self.pos}")
        
        # Epsilon-greedy exploration
        if self.random.random() < self.epsilon:
            return self.random.choice(possible_actions)
        
        # Get current policy for this state
        policy = self.get_policy(state)
        
        if not policy:
            # No policy yet, choose randomly
            return self.random.choice(possible_actions)
        
        # Sample action according to policy (mixed strategy)
        actions = list(policy.keys())
        probabilities = list(policy.values())
        
        # Ensure probabilities sum to 1
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p/total_prob for p in probabilities]
            return self.random.choices(actions, weights=probabilities)[0]
        else:
            return self.random.choice(possible_actions)
    def compute_minimax_value_and_policy(self, state):
        """
        Compute minimax value and optimal policy for a state.
        
        This is the core of Minimax-Q: solve a matrix game to find
        the minimax value and corresponding mixed strategy.
        
        Returns:
            Tuple[float, Dict]: (minimax_value, optimal_policy)
        """
        my_actions = self.get_possible_actions()
        other_positions = self.get_other_positions(state)
        
        if not my_actions:
            return 0.0, {}
        
        if not other_positions:
            # No opponents, just choose greedily
            best_action = my_actions[0]
            best_value = self.get_q_value(state, best_action, None)
            for action in my_actions[1:]:
                value = self.get_q_value(state, action, None)
                if (self.is_maximizing_player() and value > best_value) or \
                   (not self.is_maximizing_player() and value < best_value):
                    best_action = action
                    best_value = value
            return best_value, {best_action: 1.0}
        
        # For simplicity, assume single opponent
        other_pos = other_positions[0]
        # Get opponent actions using the same movement constraints (no stay action)
        other_actions = self.model.grid.get_neighborhood(other_pos, moore=True, include_center=False)
        
        # Additional safety check: ensure no stay actions for opponent
        if other_pos in other_actions:
            other_actions.remove(other_pos)
        
        if not other_actions:
            # No opponent actions, choose greedily
            best_action = my_actions[0]
            best_value = self.get_q_value(state, best_action, None)
            for action in my_actions[1:]:
                value = self.get_q_value(state, action, None)
                if (self.is_maximizing_player() and value > best_value) or \
                   (not self.is_maximizing_player() and value < best_value):
                    best_action = action
                    best_value = value
            return best_value, {best_action: 1.0}
        
        # Build payoff matrix
        payoff_matrix = []
        for my_action in my_actions:
            row = []
            for other_action in other_actions:
                q_value = self.get_q_value(state, my_action, other_action)
                row.append(q_value)
            payoff_matrix.append(row)
        
        # Solve minimax game (simplified version)
        if self.is_maximizing_player():
            # Find maximin strategy
            minimax_value, policy = self._solve_maximin(my_actions, other_actions, payoff_matrix)
        else:
            # Find minimax strategy (minimize maximum opponent can achieve)
            minimax_value, policy = self._solve_minimax(my_actions, other_actions, payoff_matrix)
        
        return minimax_value, policy
    
    def _solve_maximin(self, my_actions, other_actions, payoff_matrix):
        """
        Solve maximin problem: max_i min_j Q(s, a_i, a_j)
        
        Returns the maximin value and corresponding mixed strategy.
        For simplicity, uses pure strategy selection.
        """
        best_value = float('-inf')
        best_action = my_actions[0]
        
        for i, my_action in enumerate(my_actions):
            # Find minimum over opponent actions
            min_value = min(payoff_matrix[i])
            if min_value > best_value:
                best_value = min_value
                best_action = my_action
        
        return best_value, {best_action: 1.0}
    
    def _solve_minimax(self, my_actions, other_actions, payoff_matrix):
        """
        Solve minimax problem: min_i max_j Q(s, a_i, a_j)
        
        Returns the minimax value and corresponding mixed strategy.
        For simplicity, uses pure strategy selection.
        """
        best_value = float('inf')
        best_action = my_actions[0]
        
        for i, my_action in enumerate(my_actions):
            # Find maximum over opponent actions
            max_value = max(payoff_matrix[i])
            if max_value < best_value:
                best_value = max_value
                best_action = my_action
        
        return best_value, {best_action: 1.0}
    
    def get_policy(self, state):
        """Get current policy for state"""
        return self.policy.get(state, {})
    def set_policy(self, state, policy):
        """Set policy for state"""
        self.policy[state] = policy
    
    def get_value(self, state):
        """Get value function V(s)"""
        return self.V.get(state, 0.0)
    
    def set_value(self, state, value):
        """Set value function V(s)"""
        self.V[state] = value
    
    def update_q_value(self, state, action, other_action, reward, next_state):
        """
        Update Q-value using Minimax-Q learning rule.
        
        The Minimax-Q update is:
        Q(s,a,o) <- Q(s,a,o) + α[r + γ * V(s') - Q(s,a,o)]
        
        where V(s') is the minimax value of the next state.
        """
        current_q = self.get_q_value(state, action, other_action)
        
        if next_state is None:
            # Terminal state
            target = reward
        else:
            # Compute minimax value for next state
            next_value = self.get_value(next_state)
            if next_value == 0.0:  # Not computed yet
                next_value, next_policy = self.compute_minimax_value_and_policy(next_state)
                self.set_value(next_state, next_value)
                self.set_policy(next_state, next_policy)
            
            target = reward + self.gamma * next_value
        
        # Q-learning update
        new_q = current_q + self.alpha * (target - current_q)
        self.set_q_value(state, action, other_action, new_q)
        
        # Recompute minimax value and policy for current state
        # since Q-values have changed
        minimax_value, optimal_policy = self.compute_minimax_value_and_policy(state)
        self.set_value(state, minimax_value)
        self.set_policy(state, optimal_policy)
        
        # Debug logging
        logger.debug(f"{self.__class__.__name__} {self.unique_id}: Q-update")
        logger.debug(f"  State: {state}")
        logger.debug(f"  Action: {action}, Other: {other_action}")
        logger.debug(f"  Reward: {reward}, Q: {current_q:.3f} -> {new_q:.3f}")
    
    def step(self):
        """Execute one step of Minimax Q-Learning"""
        # Get current state
        current_state = self.get_state()
        
        # Select action using epsilon-greedy minimax
        action = self.select_action(current_state)
        
        # Store state-action for learning
        self.last_state = current_state
        self.last_action = action
        
        # Move to selected position
        if action and action != self.pos:
            self.model.grid.move_agent(self, action)
            logger.info(f"{self.__class__.__name__} {self.unique_id} moved to {action}")
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps_taken += 1
    
    def learn_from_experience(self, reward, other_action):
        """
        Learn from the last action taken.
        Called after environment provides reward and other agent's action.
        """
        if self.last_state is not None and self.last_action is not None:
            current_state = self.get_state()
            
            # Update Q-value using minimax backup
            self.update_q_value(
                self.last_state, 
                self.last_action, 
                other_action,
                reward, 
                current_state
            )
            
            self.total_reward += reward
            self._step_reward = reward
    
    def get_metrics(self):
        """Get performance metrics"""
        metrics = super().get_metrics()
        metrics.update({
            'q_table_size': len(self.Q),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / max(1, self.steps_taken),
            'steps_taken': self.steps_taken,
            'last_reward': self._step_reward
        })
        return metrics
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def get_state(self):
        """Get current state representation"""
        pass
    
    @abstractmethod
    def get_other_positions(self, state):
        """Get positions of other agents from state"""
        pass
    
    @abstractmethod
    def is_maximizing_player(self):
        """Return True if this agent is maximizing, False if minimizing"""
        pass
    
    @abstractmethod
    def calculate_reward(self):
        """Calculate immediate reward for current situation"""
        pass
    
    def choose_nash_q_action(self):
        """Choose action during Nash Q-Learning Phase 1: Action Selection (compatibility method)."""
        # Store current state for later use
        if not hasattr(self, 'observed_state'):
            self.observed_state = self.get_state()
        
        # Use minimax action selection (same as select_action but store state)
        action = self.select_action(self.observed_state)
        return action
    
    def update_q_nash(self, last_state, last_action, reward, current_state, other_action):
        """
        Update Q-table using Minimax Q-Learning (compatibility with Nash Q interface).
        
        Args:
            last_state: Previous state
            last_action: Action taken in last_state
            reward: Reward received
            current_state: Current state (next_state for Q-learning)
            other_action: Other agent's action (can be None)
        """
        if last_state is not None and last_action is not None:
            # Use the standard minimax Q-learning update
            self.update_q_value(last_state, last_action, other_action, reward, current_state)
            
            # Store for visualization
            self._step_reward = reward
            return True
        return False
