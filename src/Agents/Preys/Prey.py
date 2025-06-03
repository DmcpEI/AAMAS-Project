from mesa import Agent, Model
import logging
from Agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)
class Prey(BaseAgent):
    def step(self):
        """Single step execution: move randomly."""
        self.random_move()

    def die(self):
        self.model.grid.remove_agent(self)
        super().remove()   # calls Agent.remove(), which deregisters me