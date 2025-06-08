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
        # Hunt is handled by centralized _check_and_perform_hunting() to avoid timing issues
        # self.hunt()

