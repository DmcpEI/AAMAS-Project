import abc
import logging
from mesa import Agent, Model

logger = logging.getLogger(__name__)

class BaseAgent(Agent):
    total_kills = 0

    def __init__(self, model: Model, move_cost=1):
        super().__init__(model)
        self.move_cost = move_cost
        self.kills = 0

    def increment_kills(self):
        self.kills += 1
        type(self).total_kills += 1

    def random_move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        new_pos = self.random.choice(neighbors)
        self.model.grid.move_agent(self, new_pos)
        logger.debug(f"{self.__class__.__name__} {self.unique_id} moved to {new_pos}")

    def step(self):
        self.random_move()