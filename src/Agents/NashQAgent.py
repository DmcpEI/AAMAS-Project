import numpy as np
from Agents.BaseAgent import BaseAgent
from Agents.NashSolver import NashSolver

class NashQAgent(BaseAgent):
    """
    Base class for Nash Q-Learning agents.
    Implements Nash Q-Learning algorithm with Nash equilibrium computation.
    """    
    def __init__(self, model, alpha=0.05, gamma=0.9, epsilon=0.15, move_cost=1):
        super().__init__(model, move_cost)
        self.alpha = alpha  # learning rate (reduced for larger action space)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate (increased for larger action space)
        self.epsilon_min = 0.08  # minimum exploration rate (increased)
        self.epsilon_decay = 0.998  # decay rate per step (slower decay)
        self.Q = {}  # Q-table: (state, my_action, other_action) -> value
        self.last_state = None
        self.last_action = None
        self.last_other_action = None
        self.last_reward = 0  # Store previous reward for Q-update
          # Enhanced exploration parameters
        self.position_history = []
        self.max_history = 15  # Increased history tracking
        self.loop_detection_threshold = 3  # If same position appears 3+ times, increase exploration
        self.exploration_boost_steps = 0  # Steps remaining with boosted exploration
        
        # Q-value initialization strategy
        self.use_optimistic_initialization = True
        self.optimistic_init_value = 0.5  # Small positive initial value to encourage exploration

    def _get_q_value(self, state, action, other_action):
        """Get Q-value with optimistic initialization if enabled."""
        key = (state, action, other_action)
        if key not in self.Q and self.use_optimistic_initialization:
            # Initialize with small positive value to encourage exploration
            # Add small random component to break ties
            init_value = self.optimistic_init_value + np.random.uniform(-0.1, 0.1)
            self.Q[key] = init_value
            return init_value
        return self.Q.get(key, 0.0)      
    def _is_corner_position(self, pos):
        """Check if position is in a corner of the grid."""
        if pos is None:
            return False
        x, y = pos
        max_x = self.model.grid.width - 1
        max_y = self.model.grid.height - 1
        return (x == 0 or x == max_x) and (y == 0 or y == max_y)


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
        
        # Update position history for loop detection
        current_pos = self.pos
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Enhanced exploration logic
        current_epsilon = self.epsilon
        
        # Apply exploration boost if active
        if self.exploration_boost_steps > 0:
            current_epsilon = min(0.4, current_epsilon * 1.5)
            self.exploration_boost_steps -= 1
        
        # Detect loops and increase exploration if stuck
        if len(self.position_history) >= self.max_history:
            position_counts = {}
            for pos in self.position_history:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            max_count = max(position_counts.values())
            if max_count >= self.loop_detection_threshold:
                # Increase exploration when stuck in a loop
                current_epsilon = min(0.4, current_epsilon * 2.0)
                self.exploration_boost_steps = max(self.exploration_boost_steps, 8)
        
        # ε-greedy exploration
        if np.random.rand() < current_epsilon:
            # Random exploration - select from all available actions
            return self.random.choice(actions)
        
        # Get other agent's Q-table for Nash computation
        other_q_table = self.get_other_agent_q_table()
        
        if other_action is not None:
            # If we know the other agent's action, use Q-values with optimistic init
            q_values = [self._get_q_value(state, a, other_action) for a in actions]
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
            else:
                # Fallback to max strategy with optimistic initialization
                q_values = [max([self._get_q_value(state, a, oa) for oa in other_actions]) 
                           for a in actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
                return self.random.choice(best_actions)

    def update_epsilon(self):
        """Decay epsilon over time but maintain minimum exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_exploration_if_converged(self):
        """Reset exploration if agent seems to have converged to avoid local optima."""
        # If epsilon is at minimum and agent seems stuck, temporarily increase exploration
        if self.epsilon <= self.epsilon_min * 1.1:  # Near minimum
            # Check if we're in a repetitive pattern
            if len(self.position_history) >= self.max_history:
                recent_positions = set(self.position_history[-5:])  # Last 5 positions
                if len(recent_positions) <= 2:  # Only moving between 2 positions
                    self.epsilon = min(0.4, self.epsilon_min * 3)  # Boost exploration
                    # logger.debug(f"{self.__class__.__name__} {self.unique_id}: Convergence detected, boosting exploration to {self.epsilon:.3f}")

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
        if last_state is not None and last_action is not None:            # Get actions for Nash equilibrium computation
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
                # Fallback to max value strategy with optimistic initialization
                next_qs = [self._get_q_value(current_state, a, oa) 
                          for a in current_actions for oa in other_actions]
                best_next_q = max(next_qs) if next_qs else 0.0
            
            # Use None as placeholder if other_action is unknown
            key_other_action = other_action
            
            # Standard Q-learning update with Nash value and optimistic initialization
            old_q = self._get_q_value(last_state, last_action, key_other_action)
            new_q = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
            self.Q[(last_state, last_action, key_other_action)] = new_q
            #print(f"Updated Q-value for {self.__class__.__name__} {self.unique_id}: ")
            return new_q
        return 0.0

    def update_q_terminal(self, last_state, last_action, reward, other_action):
        """
        Update Q-table for terminal states (kills/deaths) where no next state exists.
        Terminal states have no future rewards, so future Q-value is 0.
        
        Args:
            last_state: State before terminal event
            last_action: Action that led to terminal event
            reward: Terminal reward (positive for kills, negative for deaths)
            other_action: Other agent's action (can be None)
        """
        if last_state is not None and last_action is not None:
            # Use None as placeholder if other_action is unknown
            key_other_action = other_action
            
            # Terminal state update: no future rewards (next_q = 0)
            old_q = self._get_q_value(last_state, last_action, key_other_action)
            new_q = old_q + self.alpha * (reward - old_q)  # No gamma * next_q term
            self.Q[(last_state, last_action, key_other_action)] = new_q
            
            # Debug output for terminal updates
            agent_type = "Hunter" if self.__class__.__name__.endswith("Hunter") else "Prey"
            terminal_type = "KILL" if reward > 0 else "DEATH"
            print(f"{agent_type} {self.unique_id} TERMINAL Q-update ({terminal_type}): "
                  f"Q({last_state}, {last_action}, {key_other_action}) = {new_q:.3f} "
                  f"(reward={reward}, old_q={old_q:.3f})")
            
            return new_q
        return 0.0

    def step(self):
        """Step method to be implemented by subclass."""
        raise NotImplementedError
