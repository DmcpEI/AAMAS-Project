import logging
from Agents.BaseAgent import BaseAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class RandomHunter(BaseAgent):

    total_kills = 0

    def step(self):
        """Single step execution: move and hunt."""
        # Move first        
        super().step()  # This calls random_move from BaseAgent
        # Then hunt using shared method
        self.hunt()

