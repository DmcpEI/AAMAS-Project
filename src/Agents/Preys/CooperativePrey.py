import logging
import random
import math
from Agents.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)

class CooperativePrey(BaseAgent):
    def __init__(self, model, move_cost=1):
        super().__init__(model, move_cost)
        self.shared_hunter_locations = set()  # Simple cooperation - share hunter locations
        self.role = None  # Will be assigned as 'bait' or 'escape'
        self.bait_id = None  # Track which prey is the bait
        self.strategy_cooldown = 0  # Prevent frequent strategy changes
        
    def step(self):
        """Single step execution: share info, assign roles, and move."""
        # Clear shared information at start of each step to prevent false positives
        self.shared_hunter_locations.clear()
        self.share_hunter_info()
        
        # Check strategies in order of priority using agent's own strategy flags
        if getattr(self, 'use_bait_and_switch', False):
            self.bait_and_switch_move()
        elif getattr(self, 'use_flocking', False):
            self.flocking_move()
        elif getattr(self, 'use_enhanced_escape', False):
            self.enhanced_escape_move()
        else:
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
    
    def bait_and_switch_move(self):
        """Execute Bait and Switch strategy coordinated movement."""
        # Get all cooperative prey agents
        coop_preys = [agent for agent in self.model.agents if isinstance(agent, CooperativePrey)]
        
        if len(coop_preys) < 2:
            # Need at least 2 prey for bait and switch
            self.cooperative_move()
            return
        
        # Assign roles if not already assigned or cooldown expired
        if self.strategy_cooldown <= 0:
            self.assign_bait_and_switch_roles(coop_preys)
            self.strategy_cooldown = 5  # Reset cooldown
        else:
            self.strategy_cooldown -= 1
        
        # Execute role-based movement
        if self.role == 'bait':
            self.execute_bait_behavior()
        else:
            self.execute_escape_behavior()
    
    def assign_bait_and_switch_roles(self, coop_preys):
        """Assign bait and escape roles to cooperative prey."""
        # Sort by unique_id to ensure consistent role assignment
        sorted_preys = sorted(coop_preys, key=lambda x: x.unique_id)
        
        # Find hunters and determine threat levels
        hunters = [agent for agent in self.model.agents if agent.__class__.__name__.endswith("Hunter")]
        
        if not hunters:
            # No hunters present, all prey can move freely
            for prey in sorted_preys:
                prey.role = 'escape'
            return
        
        # Calculate threat level for each prey (distance to closest hunter)
        prey_threats = []
        for prey in sorted_preys:
            min_distance = min(self.manhattan_distance(prey.pos, hunter.pos) for hunter in hunters)
            prey_threats.append((prey, min_distance))
        
        # Sort by threat level (closest to hunter first)
        prey_threats.sort(key=lambda x: x[1])
        
        # Assign roles: most threatened becomes bait, others escape
        if len(prey_threats) > 0:
            bait_prey = prey_threats[0][0]  # Most threatened
            self.bait_id = bait_prey.unique_id
            
            for prey in sorted_preys:
                if prey.unique_id == self.bait_id:
                    prey.role = 'bait'
                    logger.debug(f"Prey {prey.unique_id} assigned as BAIT")
                else:
                    prey.role = 'escape'
                    logger.debug(f"Prey {prey.unique_id} assigned as ESCAPE")
    
    def execute_bait_behavior(self):
        """Execute bait behavior - attract hunters while staying alive."""
        hunters = [agent for agent in self.model.agents if agent.__class__.__name__.endswith("Hunter")]
        
        if not hunters:
            self.random_move()
            return
        
        # Find closest hunter
        closest_hunter = min(hunters, key=lambda h: self.manhattan_distance(self.pos, h.pos))
        distance_to_hunter = self.manhattan_distance(self.pos, closest_hunter.pos)
        
        # Bait strategy: stay close enough to attract but far enough to survive
        target_distance = 2  # Optimal bait distance
        
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        best_move = self.pos
        best_score = float('-inf')
        
        for neighbor_pos in neighbors:
            if not self.model.grid.is_cell_empty(neighbor_pos):
                continue  # Can't move to occupied cell
            
            # Calculate new distance to hunter
            new_distance = self.manhattan_distance(neighbor_pos, closest_hunter.pos)
            
            # Score based on maintaining target distance
            distance_score = -abs(new_distance - target_distance)
            
            # Bonus for staying alive (avoid being too close)
            survival_score = 0 if new_distance > 1 else -100
            
            # Bonus for drawing hunter away from other prey
            distraction_score = self.calculate_distraction_value(neighbor_pos, hunters)
            
            total_score = distance_score + survival_score + distraction_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = neighbor_pos
        
        if best_move != self.pos:
            self.model.grid.move_agent(self, best_move)
            logger.debug(f"Bait prey {self.unique_id} moved to {best_move}, distance to hunter: {self.manhattan_distance(best_move, closest_hunter.pos)}")
        else:
            self.random_move()
    
    def execute_escape_behavior(self):
        """Execute escape behavior - move away from hunters and help bait."""
        hunters = [agent for agent in self.model.agents if agent.__class__.__name__.endswith("Hunter")]
        
        if not hunters:
            self.random_move()
            return
        
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        best_move = self.pos
        best_score = float('-inf')
        
        for neighbor_pos in neighbors:
            if not self.model.grid.is_cell_empty(neighbor_pos):
                continue  # Can't move to occupied cell
            
            # Primary goal: maximize distance from all hunters
            min_hunter_distance = min(self.manhattan_distance(neighbor_pos, hunter.pos) for hunter in hunters)
            escape_score = min_hunter_distance * 10
            
            # Secondary goal: stay somewhat close to help bait if needed
            support_score = self.calculate_bait_support_value(neighbor_pos)
            
            # Tertiary goal: explore toward corners/edges for better escape routes
            exploration_score = self.calculate_exploration_value(neighbor_pos)
            
            total_score = escape_score + support_score + exploration_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = neighbor_pos
        
        if best_move != self.pos:
            self.model.grid.move_agent(self, best_move)
            logger.debug(f"Escape prey {self.unique_id} moved to {best_move} for better escape position")
        else:
            self.random_move()
    
    def calculate_distraction_value(self, position, hunters):
        """Calculate how well this position draws hunters away from other prey."""
        other_preys = [agent for agent in self.model.agents 
                      if isinstance(agent, CooperativePrey) and agent != self]
        
        if not other_preys:
            return 0
        
        distraction_value = 0
        for hunter in hunters:
            # Distance from hunter to this position
            hunter_to_bait = self.manhattan_distance(hunter.pos, position)
            
            # Average distance from hunter to other prey
            avg_hunter_to_prey = sum(self.manhattan_distance(hunter.pos, prey.pos) 
                                   for prey in other_preys) / len(other_preys)
            
            # Positive value if we're closer to hunter than average (good distraction)
            if hunter_to_bait < avg_hunter_to_prey:
                distraction_value += (avg_hunter_to_prey - hunter_to_bait)
        
        return distraction_value
    
    def calculate_bait_support_value(self, position):
        """Calculate value of supporting the bait prey."""
        if not self.bait_id:
            return 0
        
        # Find bait prey
        bait_prey = None
        for agent in self.model.agents:
            if isinstance(agent, CooperativePrey) and agent.unique_id == self.bait_id:
                bait_prey = agent
                break
        
        if not bait_prey:
            return 0
        
        # Moderate distance to bait is good (close enough to help, far enough to not interfere)
        distance_to_bait = self.manhattan_distance(position, bait_prey.pos)
        optimal_support_distance = 3
        
        # Return higher score for positions near optimal support distance
        return max(0, 5 - abs(distance_to_bait - optimal_support_distance))
    
    def calculate_exploration_value(self, position):
        """Calculate value of exploring toward safer areas (corners, edges)."""
        x, y = position
        width, height = self.model.grid.width, self.model.grid.height
        
        # Distance to nearest edge
        edge_distance = min(x, y, width - 1 - x, height - 1 - y)
        
        # Prefer positions closer to edges (better escape routes)
        return max(0, 3 - edge_distance)
    
    def flocking_move(self):
        """Execute flocking behavior - stay together while avoiding hunters."""
        # Get other cooperative prey for flocking
        other_preys = [agent for agent in self.model.agents 
                      if isinstance(agent, CooperativePrey) and agent != self]
        hunters = [agent for agent in self.model.agents if agent.__class__.__name__.endswith("Hunter")]
        
        if not other_preys:
            # No other prey to flock with, use enhanced escape
            self.enhanced_escape_move()
            return
        
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        best_move = self.pos
        best_score = float('-inf')
        
        for neighbor_pos in neighbors:
            if not self.model.grid.is_cell_empty(neighbor_pos):
                continue
            
            # Flocking rules: separation, alignment, cohesion
            separation_score = self.calculate_separation_score(neighbor_pos, other_preys)
            cohesion_score = self.calculate_cohesion_score(neighbor_pos, other_preys)
            
            # Hunter avoidance (highest priority)
            hunter_avoidance_score = 0
            if hunters:
                min_hunter_distance = min(self.manhattan_distance(neighbor_pos, hunter.pos) for hunter in hunters)
                hunter_avoidance_score = min_hunter_distance * 20  # High weight for safety
            
            # Strategic positioning: prefer locations that are not too far from the center of the grid
            strategic_positioning_score = self.calculate_strategic_position_score(neighbor_pos)
            
            total_score = separation_score + cohesion_score + hunter_avoidance_score + strategic_positioning_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = neighbor_pos
        
        if best_move != self.pos:
            self.model.grid.move_agent(self, best_move)
            logger.debug(f"Flocking prey {self.unique_id} moved to {best_move}")
        else:
            self.random_move()
    
    def calculate_separation_score(self, position, other_preys):
        """Calculate separation score to avoid overcrowding."""
        separation_score = 0
        for prey in other_preys:
            distance = self.manhattan_distance(position, prey.pos)
            if distance < 2:  # Too close
                separation_score -= (2 - distance) * 5
        return separation_score
    
    def calculate_cohesion_score(self, position, other_preys):
        """Calculate cohesion score to stay with the group."""
        if not other_preys:
            return 0
        
        # Calculate center of mass of other prey
        center_x = sum(prey.pos[0] for prey in other_preys) / len(other_preys)
        center_y = sum(prey.pos[1] for prey in other_preys) / len(other_preys)
        center = (center_x, center_y)
        
        # Distance to center of mass
        distance_to_center = self.manhattan_distance(position, center)
        
        # Prefer moderate distance to center (not too far, not too close)
        optimal_distance = 2
        return max(0, 5 - abs(distance_to_center - optimal_distance))
    
    def enhanced_escape_move(self):
        """Execute enhanced escape behavior with anticipation."""
        hunters = [agent for agent in self.model.agents if agent.__class__.__name__.endswith("Hunter")]
        
        if not hunters:
            self.random_move()
            return
        
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
        best_move = self.pos
        best_score = float('-inf')
        
        for neighbor_pos in neighbors:
            if not self.model.grid.is_cell_empty(neighbor_pos):
                continue
            
            # Enhanced escape considers hunter movement patterns
            escape_score = self.calculate_anticipatory_escape_score(neighbor_pos, hunters)
            
            # Bonus for moving toward maze-like areas (more walls/obstacles)
            maze_score = self.calculate_maze_advantage_score(neighbor_pos)
            
            # Bonus for strategic positioning (near escape routes)
            strategic_score = self.calculate_strategic_position_score(neighbor_pos)
            
            total_score = escape_score + maze_score + strategic_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = neighbor_pos
        
        if best_move != self.pos:
            self.model.grid.move_agent(self, best_move)
            logger.debug(f"Enhanced escape prey {self.unique_id} moved to {best_move}")
        else:
            self.random_move()
    
    def calculate_anticipatory_escape_score(self, position, hunters):
        """Calculate escape score considering hunter movement prediction."""
        total_score = 0
        
        for hunter in hunters:
            current_distance = self.manhattan_distance(position, hunter.pos)
            
            # Predict hunter's next position (assume they move toward us)
            hunter_next_positions = self.model.grid.get_neighborhood(hunter.pos, moore=True, include_center=False)
            
            # Calculate minimum distance after hunter's potential moves
            min_future_distance = current_distance
            for hunter_next_pos in hunter_next_positions:
                future_distance = self.manhattan_distance(position, hunter_next_pos)
                min_future_distance = min(min_future_distance, future_distance)
            
            # Score based on worst-case scenario
            total_score += min_future_distance * 10
        
        return total_score
    
    def calculate_maze_advantage_score(self, position):
        """Calculate advantage of moving into maze-like areas."""
        # Count blocked cells around position (walls give tactical advantage)
        neighborhood = self.model.grid.get_neighborhood(position, moore=True, include_center=False)
        blocked_count = 0
        
        for neighbor in neighborhood:
            if not self.model.grid.is_cell_empty(neighbor):
                blocked_count += 1
        
        # Moderate blockage is good (escape routes), too much is bad (trapped)
        if blocked_count >= 6:  # Too trapped
            return -10
        elif blocked_count >= 3:  # Good tactical advantage
            return blocked_count * 2
        else:  # Open area, vulnerable
            return 0
    
    def calculate_strategic_position_score(self, position):
        """Calculate strategic positioning score."""
        x, y = position
        width, height = self.model.grid.width, self.model.grid.height
        
        # Distance to corners (good escape positions)
        corner_distances = [
            self.manhattan_distance(position, (0, 0)),
            self.manhattan_distance(position, (0, height-1)),
            self.manhattan_distance(position, (width-1, 0)),
            self.manhattan_distance(position, (width-1, height-1))
        ]
        
        min_corner_distance = min(corner_distances)
        
        # Prefer being somewhat close to corners but not trapped
        if min_corner_distance <= 2:
            return 5
        elif min_corner_distance <= 4:
            return 3
        else:
            return 0
    
    def cooperative_move(self):
        """Move away from hunters using shared information."""
        # Find dangerous hunters (close to us)
        dangerous_hunters = []
        for hunter_pos in self.shared_hunter_locations:
            if self.manhattan_distance(self.pos, hunter_pos) <= 2:  # Danger threshold
                dangerous_hunters.append(hunter_pos)
        
        if dangerous_hunters:
            # Try to move away from hunters
            neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
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