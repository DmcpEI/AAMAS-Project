import abc
import logging
from mesa import Agent, Model

logger = logging.getLogger(__name__)

class BaseAgent(Agent):
    total_kills = 0

    def __init__(self, model: Model, energy=20, move_cost=1):
        super().__init__(model)
        self.energy = energy
        self.move_cost = move_cost
        self.kills = 0

    def increment_kills(self):
        self.kills += 1
        type(self).total_kills += 1

    def random_move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_pos = self.random.choice(neighbors)
        self.model.grid.move_agent(self, new_pos)
        logger.debug(f"{self.__class__.__name__} {self.unique_id} moved to {new_pos}")

    def step(self):
        if self.energy > 0:
            self.energy -= self.move_cost
            self.random_move()