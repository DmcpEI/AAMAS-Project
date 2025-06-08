import logging
from Agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)

class CooperativePrey(BaseAgent):
    
    def __init__(self, model, move_cost=1):
        super().__init__(model, move_cost)
        self.shared_hunter_locations = set()  # Simple cooperation - share hunter locations
    
    def step(self):
        """Single step execution: share info and move."""
        self.share_hunter_info()
        self.cooperative_move()
    
    def share_hunter_info(self):
        """Share hunter locations with other cooperative prey."""
        # Find all hunters
        hunters = [agent for agent in self.model.agents if agent.__class__.__name__.endswith("Hunter")]
        
        # Add hunter positions to shared knowledge
        for hunter in hunters:
            self.shared_hunter_locations.add(hunter.pos)
        
        # Share with other cooperative prey
        for agent in self.model.agents:
            if isinstance(agent, CooperativePrey) and agent != self:
                # Share our knowledge with them
                agent.shared_hunter_locations.update(self.shared_hunter_locations)
                # Learn from their knowledge
                self.shared_hunter_locations.update(agent.shared_hunter_locations)
    
    def cooperative_move(self):
        """Move away from hunters using shared information."""
        # Find dangerous hunters (close to us)
        dangerous_hunters = []
        for hunter_pos in self.shared_hunter_locations:
            if self.manhattan_distance(self.pos, hunter_pos) <= 2:  # Danger threshold
                dangerous_hunters.append(hunter_pos)
        
        if dangerous_hunters:
            # Try to move away from hunters
            neighbors = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
            best_move = self.pos
            best_safety = -1
            
            for neighbor_pos in neighbors:
                # Calculate minimum distance to any dangerous hunter
                min_distance = min(self.manhattan_distance(neighbor_pos, hunter_pos) 
                                 for hunter_pos in dangerous_hunters)
                
                if min_distance > best_safety:
                    best_safety = min_distance
                    best_move = neighbor_pos
            
            if best_move != self.pos:
                self.model.grid.move_agent(self, best_move)
            else:
                self.random_move()
        else:
            # No immediate danger, move randomly
            self.random_move()
    
    def die(self):
        """Handle death (being caught by hunter)."""
        self.model.grid.remove_agent(self)
        super().remove()  # calls Agent.remove(), which deregisters from model
