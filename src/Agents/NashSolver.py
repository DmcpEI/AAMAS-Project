import numpy as np
from scipy.optimize import linprog
import itertools

class NashSolver:
    """
    Nash equilibrium solver for 2-player games.
    Implements linear programming approach to find Nash equilibria.
    """
    
    @staticmethod
    def solve_nash_equilibrium(payoff_matrix_p1, payoff_matrix_p2):
        """
        Solve for Nash equilibrium in a 2-player game.
        
        Args:
            payoff_matrix_p1: numpy array of shape (m, n) - Player 1's payoffs
            payoff_matrix_p2: numpy array of shape (m, n) - Player 2's payoffs
            
        Returns:
            tuple: (strategy_p1, strategy_p2, value_p1, value_p2)
                - strategy_p1: probability distribution over Player 1's actions
                - strategy_p2: probability distribution over Player 2's actions
                - value_p1: expected payoff for Player 1
                - value_p2: expected payoff for Player 2
        """
        m, n = payoff_matrix_p1.shape
        
        # Solve for Player 1's mixed strategy
        strategy_p1, value_p1 = NashSolver._solve_player_strategy(payoff_matrix_p1)
        
        # Solve for Player 2's mixed strategy (transpose the game)
        strategy_p2, value_p2 = NashSolver._solve_player_strategy(payoff_matrix_p2.T)
        
        return strategy_p1, strategy_p2, value_p1, value_p2
    
    @staticmethod
    def _solve_player_strategy(payoff_matrix):
        """
        Solve for optimal mixed strategy using linear programming.
        
        Args:
            payoff_matrix: numpy array of player's payoffs
            
        Returns:
            tuple: (strategy, expected_value)
        """
        m, n = payoff_matrix.shape
        
        # Linear programming formulation for mixed strategy Nash equilibrium
        # Variables: [p1, p2, ..., pm, v] where pi is probability of action i, v is value
        # Minimize -v (maximize v)
        c = np.zeros(m + 1)
        c[-1] = -1  # coefficient for v (we want to maximize v)
        
        # Inequality constraints: for each column j, sum(pi * payoff[i,j]) >= v
        A_ub = []
        b_ub = []
        
        for j in range(n):
            constraint = np.zeros(m + 1)
            constraint[:m] = -payoff_matrix[:, j]  # -sum(pi * payoff[i,j])
            constraint[-1] = 1  # +v
            A_ub.append(constraint)
            b_ub.append(0)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraint: sum of probabilities = 1
        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1  # sum(pi) = 1
        b_eq = np.array([1])
        
        # Bounds: probabilities >= 0, value unbounded
        bounds = [(0, None)] * m + [(None, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                           bounds=bounds, method='highs')
            
            if result.success:
                strategy = result.x[:m]
                value = result.x[-1]
                
                # Normalize strategy (numerical precision)
                strategy = strategy / np.sum(strategy)
                
                return strategy, value
            else:
                # Fallback to uniform strategy
                strategy = np.ones(m) / m
                value = np.mean(np.sum(payoff_matrix * strategy.reshape(-1, 1), axis=0))
                return strategy, value
                
        except Exception:
            # Fallback to uniform strategy
            strategy = np.ones(m) / m
            value = np.mean(np.sum(payoff_matrix * strategy.reshape(-1, 1), axis=0))
            return strategy, value
    
    @staticmethod
    def compute_nash_value(q_table_p1, q_table_p2, state, actions_p1, actions_p2):
        """
        Compute Nash equilibrium value for a given state using Q-tables.
        
        Args:
            q_table_p1: Player 1's Q-table (dict)
            q_table_p2: Player 2's Q-table (dict)
            state: Current state
            actions_p1: List of possible actions for Player 1
            actions_p2: List of possible actions for Player 2
            
        Returns:
            tuple: (nash_value_p1, nash_value_p2)
        """
        m, n = len(actions_p1), len(actions_p2)
        
        # Build payoff matrices from Q-tables
        payoff_matrix_p1 = np.zeros((m, n))
        payoff_matrix_p2 = np.zeros((m, n))
        
        for i, action_p1 in enumerate(actions_p1):
            for j, action_p2 in enumerate(actions_p2):
                # Q-value for player 1: Q(state, action_p1, action_p2)
                payoff_matrix_p1[i, j] = q_table_p1.get((state, action_p1, action_p2), 0.0)
                # Q-value for player 2: Q(state, action_p2, action_p1)
                payoff_matrix_p2[i, j] = q_table_p2.get((state, action_p2, action_p1), 0.0)
        
        # Solve for Nash equilibrium
        strategy_p1, strategy_p2, value_p1, value_p2 = NashSolver.solve_nash_equilibrium(
            payoff_matrix_p1, payoff_matrix_p2)
        
        return value_p1, value_p2
    @staticmethod
    def select_nash_action(q_table, state, my_actions, other_actions, other_q_table=None):
        """
        Select action based on Nash equilibrium strategy.
        
        Args:
            q_table: Agent's Q-table
            state: Current state
            my_actions: Agent's possible actions
            other_actions: Other agent's possible actions
            other_q_table: Other agent's Q-table (if available)
            
        Returns:
            Selected action based on Nash equilibrium strategy
        """
        if other_q_table is None:
            # If we don't have access to other agent's Q-table, use max strategy
            q_values = []
            for my_action in my_actions:
                max_q = max([q_table.get((state, my_action, other_action), 0.0) 
                           for other_action in other_actions])
                q_values.append(max_q)
            return my_actions[np.argmax(q_values)]
        
        # Build payoff matrices
        m, n = len(my_actions), len(other_actions)
        my_payoffs = np.zeros((m, n))
        other_payoffs = np.zeros((m, n))
        for i, my_action in enumerate(my_actions):
            for j, other_action in enumerate(other_actions):
                my_payoffs[i, j] = q_table.get((state, my_action, other_action), 0.0)
                other_payoffs[i, j] = other_q_table.get((state, other_action, my_action), 0.0)
        
        # Solve Nash equilibrium
        my_strategy, _, _, _ = NashSolver.solve_nash_equilibrium(my_payoffs, other_payoffs.T)
        
        # Sample action according to strategy using indices
        action_index = np.random.choice(len(my_actions), p=my_strategy)
        return my_actions[action_index]
