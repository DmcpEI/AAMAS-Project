import logging
from Agents.BaseAgent import BaseAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class GreedyHunter(BaseAgent):
    total_kills = 0
    
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)
    
    def step(self):
        if self.energy > 0:
            self.energy -= self.move_cost
            self.greedy_move()
            self.hunt()

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions on a non-toroidal grid"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return dx + dy

    def find_closest_prey(self):
        preys = [agent for agent in self.model.agents if isinstance(agent, Prey)]
        closest_prey = None
        closest_dist = float('inf')
        for prey in preys:
            dist = self.manhattan_distance(self.pos, prey.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_prey = prey
        return closest_prey, closest_dist

    def find_best_move_towards(self, target_pos, current_best_dist):
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        best_move = self.pos
        best_dist = current_best_dist
        for npos in neighbors:
            dist = self.manhattan_distance(npos, target_pos)
            if dist < best_dist:
                best_dist = dist
                best_move = npos
        return best_move

    def greedy_move(self):
        closest_prey, closest_prey_dist = self.find_closest_prey()
        if not closest_prey:
            self.random_move()
            return
        best_move = self.find_best_move_towards(closest_prey.pos, closest_prey_dist)
        self.model.grid.move_agent(self, best_move)
        logger.debug(f"GreedyHunter {self.unique_id} moved to {best_move} towards Prey {closest_prey.unique_id}")

    def hunt(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for other in list(cellmates):
            if isinstance(other, Prey):
                logger.info(f"GreedyHunter {self.unique_id} ate Prey {other.unique_id} at {self.pos}")
                other.die()
                self.increment_kills()