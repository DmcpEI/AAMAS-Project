import logging
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from Agents.Hunters.RandomHunter import RandomHunter
from Agents.Hunters.GreedyHunter import GreedyHunter
from Agents.Hunters.NashQHunter import NashQHunter
from Agents.Hunters.MinimaxHunter import MinimaxHunter
from Agents.Preys.Prey import Prey
from Agents.Preys.NashQPrey import NashQPrey
from Agents.Preys.MinimaxPrey import MinimaxPrey
from typing import Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class HunterPreyModel(Model):
    """
    A model with dynamic numbers of hunters and preys on a non-toroidal grid.
    Uses the AgentSet API for scheduling.
    """
    def __init__(
        self,
        N_hunters: int = 10,
        N_greedy_hunters: int = 0,        N_nash_q_hunters: int = 0,
        N_minimax_hunters: int = 0,
        N_preys: int = 50,
        N_nash_q_preys: int = 0,
        N_minimax_preys: int = 0,
        width: int = 20,
        height: int = 20,
        minimax_search_depth: int = 4,
        seed: Optional[int] = None
    ):        # Initialize base Model (sets up self.agents, RNG, etc.)
        super().__init__(seed=seed)        
        logger.info(
            f"Initializing model with {N_hunters} hunters, {N_greedy_hunters} greedy hunters, {N_nash_q_hunters} Nash Q-learning hunters, {N_minimax_hunters} minimax hunters, {N_preys} preys, {N_nash_q_preys} Nash Q-learning preys, and {N_minimax_preys} minimax preys "
            f"on a {width}x{height} grid"
        )
        self.num_hunters = N_hunters
        self.num_greedy_hunters = N_greedy_hunters
        self.num_nash_q_hunters = N_nash_q_hunters
        self.num_minimax_hunters = N_minimax_hunters
        self.num_preys = N_preys
        self.num_nash_q_preys = N_nash_q_preys
        self.num_minimax_preys = N_minimax_preys
        self.minimax_search_depth = minimax_search_depth        
        self.grid = MultiGrid(width, height, torus=False)
        
        # Kill notification system
        self.kill_occurred_this_step = False
        self.kill_info = None  # Will store {'hunter_id': X, 'prey_id': Y, 'hunter_type': 'type', 'position': (x,y)}
        self.pending_hunter_teleports = []  # List of hunters waiting to teleport next step
        
        # Reset kill counters
        RandomHunter.total_kills = 0
        GreedyHunter.total_kills = 0
        NashQHunter.total_kills = 0
        MinimaxHunter.total_kills = 0        # Data collector for live counts
        self.datacollector = DataCollector({
            "Hunters": lambda m: m.count_type(RandomHunter),
            "GreedyHunters": lambda m: m.count_type(GreedyHunter),
            "NashQHunters": lambda m: m.count_type(NashQHunter),
            "MinimaxHunters": lambda m: m.count_type(MinimaxHunter),
            "Preys": lambda m: m.count_type(Prey),
            "NashQPreys": lambda m: m.count_type(NashQPrey),
            "MinimaxPreys": lambda m: m.count_type(MinimaxPrey),
            "AvgEnergy": self.avg_energy,
            "RandomHunterKills": self.get_random_hunter_kills,
            "GreedyHunterKills": self.get_greedy_hunter_kills,
            "NashQHunterKills": self.get_nash_q_hunter_kills,
            "MinimaxHunterKills": self.get_minimax_hunter_kills,
            "AvgHunterReward": self.avg_hunter_reward,
            "AvgNashQHunterReward": self.avg_nash_q_hunter_reward,
            "AvgMinimaxHunterReward": self.avg_minimax_hunter_reward,
            "AvgPreyReward": self.avg_prey_reward,
            "AvgNashQPreyReward": self.avg_nash_q_prey_reward,
            "AvgMinimaxPreyReward": self.avg_minimax_prey_reward,
        })# Create and place hunter agents
        for _ in range(self.num_hunters):
            hunter = RandomHunter(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(hunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(hunter, (x, y))
            #logger.debug(f"Placed RandomHunter {hunter.unique_id} at {(x, y)}")

        # Create and place greedy hunter agents
        for _ in range(self.num_greedy_hunters):
            ghunter = GreedyHunter(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(ghunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(ghunter, (x, y))
            #logger.debug(f"Placed GreedyHunter {ghunter.unique_id} at {(x, y)}")        # Create and place Nash Q-learning hunter agents
        for _ in range(self.num_nash_q_hunters):
            nahunter = NashQHunter(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(nahunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(nahunter, (x, y))
            #logger.debug(f"Placed NashQHunter {nahunter.unique_id} at {(x, y)}")

        # Create and place Minimax hunter agents
        for _ in range(self.num_minimax_hunters):
            mihunter = MinimaxHunter(self, search_depth=self.minimax_search_depth)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(mihunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(mihunter, (x, y))
            #logger.debug(f"Placed MinimaxHunter {mihunter.unique_id} at {(x, y)}")

        # Create and place prey agents
        for _ in range(self.num_preys):
            prey = Prey(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(prey, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(prey, (x, y))
        for _ in range(self.num_nash_q_preys):
            naprey = NashQPrey(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(naprey, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(naprey, (x, y))            #logger.debug(f"Placed NashQPrey {naprey.unique_id} at {(x, y)}")
        
        # Create and place Minimax prey agents
        for _ in range(self.num_minimax_preys):
            miprey = MinimaxPrey(self, search_depth=self.minimax_search_depth)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(miprey, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(miprey, (x, y))
            #logger.debug(f"Placed MinimaxPrey {miprey.unique_id} at {(x, y)}")
        
        # Initial data collection
        self.running = True
        self.datacollector.collect(self)

    def _get_collision_free_position(self):
        """Get a random position that doesn't have other agents."""
        # Get all possible positions
        all_positions = [(x, y) for x in range(self.grid.width) for y in range(self.grid.height)]
          # Filter out positions that already have agents
        available_positions = []
        for pos in all_positions:
            cell_contents = self.grid.get_cell_list_contents([pos])
            # Only consider truly empty cells (no agents)
            if not cell_contents:
                available_positions.append(pos)
        if available_positions:
            return self.random.choice(available_positions)
        else:
            # Fallback to any random position if no completely empty cells available
            return None    
    def step(self) -> None:
        """Advance the model by one step."""
        logger.debug(f"Model step {self.steps}")

        # Process pending teleports from previous kills
        if self.pending_hunter_teleports:
            for hunter in list(self.pending_hunter_teleports):
                new_pos = self._get_collision_free_position()
                if new_pos:
                    self.grid.move_agent(hunter, new_pos)
            # Clear pending list
            self.pending_hunter_teleports.clear()

        # Reset kill tracking for this step
        self.kill_occurred_this_step = False
        self.kill_info = None

        # Use regular Mesa agent activation but with explicit list to avoid
        # concurrent modification issues during agent removal (when prey dies)
        agents_list = list(self.agents)
        self.random.shuffle(agents_list)  # Random activation order
        
        for agent in agents_list:
            # Check if agent still exists (not removed during this step)
            if agent in self.agents:
                agent.step()
        
        # Collect metrics
        self.datacollector.collect(self)

    def count_type(self, agent_type: type) -> int:
        """Helper to count agents of a given type."""
        count = sum(isinstance(agent, agent_type) for agent in self.agents)
        logger.debug(f"Current {agent_type.__name__} count: {count}")
        return count
    
    def avg_energy(self) -> float:
        """Compute average energy safely as a model method."""
        try:
            agents = [a for a in self.agents if hasattr(a, "energy")]
            return sum(a.energy for a in agents) / len(agents) if agents else 0.0
        except Exception as e:
            logger.warning(f"AvgEnergy failed at step {self.steps}: {e}")
            return 0.0
    
    def get_random_hunter_kills(self):
        return RandomHunter.total_kills

    def get_greedy_hunter_kills(self):
        return GreedyHunter.total_kills
    
    def get_nash_q_hunter_kills(self):
        return NashQHunter.total_kills
    
    def get_minimax_hunter_kills(self):
        return MinimaxHunter.total_kills
    
    def avg_hunter_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, RandomHunter)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_nash_q_hunter_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, NashQHunter)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_prey_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, Prey)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_nash_q_prey_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, NashQPrey)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_minimax_hunter_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, MinimaxHunter)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_minimax_prey_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, MinimaxPrey)]
        return np.mean(rewards) if rewards else 0.0      
    def register_kill(self, hunter_agent, prey_agent):
        """Register a kill event"""
        self.kill_occurred_this_step = True
        self.kill_info = {
            'hunter_id': hunter_agent.unique_id,
            'prey_id': prey_agent.unique_id,
            'hunter_type': hunter_agent.__class__.__name__,
            'prey_type': prey_agent.__class__.__name__,
            'position': hunter_agent.pos,
            'step': self.steps
        }

    