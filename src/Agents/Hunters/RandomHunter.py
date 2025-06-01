import logging
from Agents.BaseAgent import BaseAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class RandomHunter(BaseAgent):
    def move_phase(self):
        """Phase 1: Movement only."""
        super().step()  # This calls random_move

    def action_phase(self):
        """Phase 2: Actions only - hunting."""
        self.hunt()    
    def step(self):
        """Legacy step method - calls both phases for backward compatibility."""
        self.move_phase()
        self.action_phase()
    
    def hunt(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])  # get all agents in the cell
        for other in list(cellmates):
            if isinstance(other, Prey):
                logger.info(f"RandomHunter {self.unique_id} ate Prey {other.unique_id} at {self.pos}")
                # Register kill for visualization
                self.model.register_kill(self, other)
                # Remove prey from the model
                other.die()
                # Increment kill counter
                self.increment_kills()
                # Schedule teleportation next step
                self.model.pending_hunter_teleports.append(self)
                # End hunt
                break
        return False
