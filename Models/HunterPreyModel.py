import logging
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from Agents.Hunter import Hunter
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
        N_preys: int = 50,
        width: int = 20,
        height: int = 20,
        seed: Optional[int] = None
    ):
        # Initialize base Model (sets up self.agents, RNG, etc.)
        super().__init__(seed=seed)
        logger.info(
            f"Initializing model with {N_hunters} hunters, {N_preys} preys "
            f"on a {width}x{height} grid"
        )
        self.num_hunters = N_hunters
        self.num_preys = N_preys
        self.grid = MultiGrid(width, height, torus=True)

        # Data collector for live counts
        self.datacollector = DataCollector({
            "Hunters": lambda m: m.count_type(Hunter),
            "Preys": lambda m: m.count_type(Prey)
        })

        # Create and place hunter agents
        for _ in range(self.num_hunters):
            hunter = Hunter(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(hunter, (x, y))
            logger.debug(f"Placed Hunter {hunter.unique_id} at {(x, y)}")

        # Create and place prey agents
        for _ in range(self.num_preys):
            prey = Prey(self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(prey, (x, y))
            logger.debug(f"Placed Prey {prey.unique_id} at {(x, y)}")

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
