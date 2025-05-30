import logging
from Agents.BaseAgent import BaseAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class RandomHunter(BaseAgent):
    def step(self):
        # move first
        super().step()
        # search for a prey in the new cell
        self.hunt()
            
    def hunt(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos]) # get all agents in the cell
        for other in list(cellmates):
            if isinstance(other, Prey):
                logger.info(f"RandomHunter {self.unique_id} ate Prey {other.unique_id} at {self.pos}")
                # remove prey from the model
                other.die()
                self.increment_kills()
