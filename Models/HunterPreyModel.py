import logging
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from Agents.Hunters.RandomHunter import RandomHunter
from Agents.Hunters.GreedyHunter import GreedyHunter
from Agents.Prey import Prey
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class HunterPreyModel(Model):
    """
    A model with dynamic numbers of hunters and preys on a toroidal grid.
    Uses the AgentSet API for scheduling.
    """
    
    def __init__(
        self,
        N_hunters: int = 10,
        N_greedy_hunters: int = 0,
        N_preys: int = 50,
        width: int = 20,
        height: int = 20,
        seed: Optional[int] = None
    ):
        # Initialize base Model (sets up self.agents, RNG, etc.)
        super().__init__(seed=seed)
        logger.info(
            f"Initializing model with {N_hunters} hunters, {N_greedy_hunters} greedy hunters, {N_preys} preys "
            f"on a {width}x{height} grid"
        )
        self.num_hunters = N_hunters
        self.num_greedy_hunters = N_greedy_hunters
        self.num_preys = N_preys
        self.grid = MultiGrid(width, height, torus=True)

        # Reset kill counters
        RandomHunter.total_kills = 0
        GreedyHunter.total_kills = 0

        # Data collector for live counts
        self.datacollector = DataCollector({
            "Hunters": lambda m: m.count_type(RandomHunter),
            "GreedyHunters": lambda m: m.count_type(GreedyHunter),
            "Preys": lambda m: m.count_type(Prey),
            "AvgEnergy": self.avg_energy,
            "RandomHunterKills": self.get_random_hunter_kills,
            "GreedyHunterKills": self.get_greedy_hunter_kills,
        })

        # Create and place hunter agents
        for _ in range(self.num_hunters):
            hunter = RandomHunter(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(hunter, (x, y))
            #logger.debug(f"Placed RandomHunter {hunter.unique_id} at {(x, y)}")

        # Create and place greedy hunter agents
        for _ in range(self.num_greedy_hunters):
            ghunter = GreedyHunter(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(ghunter, (x, y))
            #logger.debug(f"Placed GreedyHunter {ghunter.unique_id} at {(x, y)}")

        # Create and place prey agents
        for _ in range(self.num_preys):
            prey = Prey(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(prey, (x, y))
            #logger.debug(f"Placed Prey {prey.unique_id} at {(x, y)}")

        # Initial data collection
        self.running = True
        self.datacollector.collect(self)

    def step(self) -> None:
        """Advance the model by one step: shuffle and step all agents, then collect data."""
        logger.debug(f"Model step {self.steps}")
        # Random activation of all agents
        self.agents.shuffle_do("step")
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
    
