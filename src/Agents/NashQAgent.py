import numpy as np
from Agents.BaseAgent import BaseAgent
from Agents.NashSolver import NashSolver

class NashQAgent(BaseAgent):
    """
    Base class for Nash Q-Learning agents.
    Implements Nash Q-Learning algorithm with Nash equilibrium computation.
    """
    
    def __init__(self, model, alpha=0.1, gamma=0.9, epsilon=0.1, move_cost=1):
        super().__init__(model, move_cost)
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.Q = {}  # Q-table: (state, my_action, other_action) -> value
        self.last_state = None
        self.last_action = None
        self.last_other_action = None
        self.last_reward = 0  # Store previous reward for Q-update

    def get_state(self, agent_type_filter):
        return super().get_state(agent_type_filter)

    def get_possible_other_actions(self, other_pos):
        """Get possible actions for other agent. Must be implemented by subclass."""
        raise NotImplementedError

    def get_other_positions(self, state):
        """Extract other agent positions from state. Must be implemented by subclass."""
        raise NotImplementedError
    
    def get_other_agent_q_table(self):
        """
        Get other agent's Q-table for Nash equilibrium computation.
        Must be implemented by subclass to identify the other agent.
        """
        raise NotImplementedError

    def select_action(self, state, other_action=None):
        """
        Select action using Nash Q-Learning strategy.
        """
        actions = self.get_possible_actions()
        
        # ε-greedy exploration
        if np.random.rand() < self.epsilon:
            return self.random.choice(actions)
        
        # Get other agent's Q-table for Nash computation
        other_q_table = self.get_other_agent_q_table()
        
        if other_action is not None:
            # If we know the other agent's action, use standard Q-values
            q_values = [self.Q.get((state, a, other_action), 0.0) for a in actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            return self.random.choice(best_actions)
        else:
            # Use Nash equilibrium strategy
            other_positions = self.get_other_positions(state)
            if other_positions:
                other_pos = other_positions[0]
                other_actions = self.get_possible_other_actions(other_pos)
            else:
                other_actions = [None]
            
            if other_q_table is not None and len(other_actions) > 1:
                # Use Nash equilibrium action selection
                return NashSolver.select_nash_action(
                    self.Q, state, actions, other_actions, other_q_table)
            else:                # Fallback to max strategy
                q_values = [max([self.Q.get((state, a, oa), 0.0) for oa in other_actions]) 
                           for a in actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
                return self.random.choice(best_actions)

    def update_q_nash(self, last_state, last_action, reward, current_state, other_action):
        """
        Update Q-table using Nash Q-Learning algorithm.
        Args:
            last_state: Previous state
            last_action: Action taken in last_state
            reward: Reward received
            current_state: Current state (next_state for Q-learning)
            other_action: Other agent's action (can be None)
        """
        if last_state is not None and last_action is not None:
            # Get actions for Nash equilibrium computation
            current_actions = self.get_possible_actions()
            other_positions = self.get_other_positions(current_state)
            
            if other_positions:
                other_pos = other_positions[0]
                other_actions = self.get_possible_other_actions(other_pos)
            else:
                other_actions = [None]
            
            # Get other agent's Q-table
            other_q_table = self.get_other_agent_q_table()
            
            if other_q_table is not None and len(current_actions) > 1 and len(other_actions) > 1:
                # Compute Nash equilibrium value for current state
                nash_value_me, _ = NashSolver.compute_nash_value(
                    self.Q, other_q_table, current_state, current_actions, other_actions)
                best_next_q = nash_value_me
            else:
                # Fallback to max value strategy
                next_qs = [self.Q.get((current_state, a, oa), 0.0) 
                          for a in current_actions for oa in other_actions]
                best_next_q = max(next_qs) if next_qs else 0.0
            
            # Use None as placeholder if other_action is unknown
            key_other_action = other_action if other_action is not None else "unknown"
            
            # Standard Q-learning update with Nash value
            old_q = self.Q.get((last_state, last_action, key_other_action), 0.0)
            new_q = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
            self.Q[(last_state, last_action, key_other_action)] = new_q
            
            return new_q
        return 0.0

    def step(self):
        """Step method to be implemented by subclass."""
        raise NotImplementedError
