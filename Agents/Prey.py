from mesa import Agent, Model
import logging

logger = logging.getLogger(__name__)
class Prey(Agent):
    """Prey agent that moves randomly."""
    def __init__(self, model: Model):
        super().__init__( model)

    def step(self):
        logger.debug(f"Prey {self.unique_id} stepping")
        self.random_move()

    def random_move(self):
        neighbors = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        new_pos = self.random.choice(neighbors)
        self.model.grid.move_agent(self, new_pos)
        logger.debug(f"Prey {self.unique_id} moved to {new_pos}")