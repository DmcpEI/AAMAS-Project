import abc
import logging
from mesa import Agent, Model

logger = logging.getLogger(__name__)

class BaseAgent(Agent):
    """Abstract-ish base with a default step that just moves randomly."""
    def __init__(self, model: Model):
        super().__init__(model)

    def random_move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_pos = self.random.choice(neighbors)
        self.model.grid.move_agent(self, new_pos)
        logger.debug(f"{self.__class__.__name__} {self.unique_id} moved to {new_pos}")

    def step(self):
        """Default behavior: random walk."""
        self.random_move()