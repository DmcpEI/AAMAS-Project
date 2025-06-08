# Constants
class ModelConfig:
    """Configuration constants for the Hunter-Prey model"""
    # Display and notification settings
    DEFAULT_KILL_NOTIFICATION_DURATION = 1
    DEFAULT_EXPLORATION_THRESHOLD = 3
    DEFAULT_MAX_HISTORY = 15
    
    # Reward and penalty values
    SUCCESSFUL_HUNT_REWARD = 10.0
    MOVEMENT_COST = -1
    SURVIVAL_REWARD = 1
    DEATH_PENALTY = -10.0
    
    
    # Nash Q-learning phases
    NASH_Q_PHASE_NORMAL = "normal"
    NASH_Q_PHASE_CHOOSE_ACTION = "choose_action"
    NASH_Q_PHASE_OBSERVE = "observe"
    NASH_Q_PHASE_UPDATE = "update"
      # Interaction distance (Manhattan distance)
    MAX_INTERACTION_DISTANCE = 1    # Mixed strategy configuration - Always use probabilistic action selection
    USE_MIXED_STRATEGIES = True
    RANDOM_EXPLORATION_PROBABILITY = 0.05  # 5% probability for pure random exploration (uniform distribution)