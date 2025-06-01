from mesa import Agent, Model
import logging
from Agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)
class Prey(BaseAgent):
    def move_phase(self):
        """Phase 1: Movement only."""
        self.random_move()

    def action_phase(self):
        """Phase 2: Actions (survival for basic prey - no special actions)."""
        pass  # Basic prey doesn't have special actions

    def step(self):
        """Legacy step method - calls both phases for backward compatibility."""
        self.move_phase()
        self.action_phase()

    def die(self):
        self.model.grid.remove_agent(self)
        super().remove()   # calls Agent.remove(), which deregisters me