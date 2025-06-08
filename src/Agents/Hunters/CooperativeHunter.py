import logging
import math
from Agents.BaseAgent import BaseAgent
from Agents.Preys.Prey import Prey

logger = logging.getLogger(__name__)

class CooperativeHunter(BaseAgent):
    total_kills = 0
    
    def __init__(self, model, move_cost=1):
        super().__init__(model, move_cost)
        self.shared_prey_locations = set()  # Simple cooperation - share prey locations
        self.hunter_id = self.unique_id  # For coordination purposes
        self.preferred_approach_angle = None  # Will be assigned based on hunter ID
        
        # Strategy selection attributes
        self.current_strategy = "direct"  # Options: "direct", "flanking", "formation"
        self.formation_role = None  # For formation hunting: "leader", "flanker", "blocker"
        self.formation_target = None  # Which prey this formation is targeting
        
        # Available strategies - will be set by model after creation
        self.available_strategies = ["direct"]  # Default to only direct pursuit
    
    def step(self):
        """Single step execution: share info, move and hunt."""
        self.share_prey_info()
        self.cooperative_move()
        self.hunt()
    
    def share_prey_info(self):
        """Share prey locations with other cooperative hunters."""
        # Find all prey
        preys = [agent for agent in self.model.agents if isinstance(agent, Prey) or agent.__class__.__name__.endswith("Prey")]
        
        # Add prey positions to shared knowledge
        for prey in preys:
            self.shared_prey_locations.add(prey.pos)
        
        # Share with other cooperative hunters
        for agent in self.model.agents:
            if isinstance(agent, CooperativeHunter) and agent != self:
                # Share our knowledge with them
                agent.shared_prey_locations.update(self.shared_prey_locations)
                # Learn from their knowledge
                self.shared_prey_locations.update(agent.shared_prey_locations)
    
    def find_closest_prey(self):
        """Find closest prey using shared information."""
        # First try direct observation
        preys = [agent for agent in self.model.agents if isinstance(agent, Prey) or agent.__class__.__name__.endswith("Prey")]
        
        if preys:
            closest_prey = None
            closest_dist = float('inf')
            for prey in preys:
                dist = self.manhattan_distance(self.pos, prey.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_prey = prey
            return closest_prey, closest_dist
        
        # If no direct prey found, use shared information
        if self.shared_prey_locations:
            closest_pos = min(self.shared_prey_locations, 
                            key=lambda pos: self.manhattan_distance(self.pos, pos))
            return None, self.manhattan_distance(self.pos, closest_pos)
        return None, float('inf')
    
    def find_best_move_towards(self, target_pos, current_best_dist):
        """Same as GreedyHunter - find best move towards target."""
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        best_move = self.pos  # Initialize to current position (stay action)
        best_dist = current_best_dist
        
        for npos in neighbors:
            dist = self.manhattan_distance(npos, target_pos)
            if dist < best_dist:
                best_dist = dist
                best_move = npos
        return best_move

    def cooperative_move(self):
        """Move towards prey using shared information with intelligent strategy selection."""
        closest_prey, closest_prey_dist = self.find_closest_prey()
        
        if closest_prey:
            # Get available strategies from model settings (respects checkboxes)
            available_strategies = self.model.get_available_strategies()
            logger.info(f"CooperativeHunter {self.unique_id}: Available strategies: {available_strategies} "
                       f"(model settings: flanking={self.model.use_flanking}, formation={self.model.use_formation_hunting})")
            
            # Get other hunters for strategy analysis
            other_hunters = [agent for agent in self.model.agents 
                           if isinstance(agent, CooperativeHunter) and agent != self]
            
            # Intelligent strategy selection
            selected_strategy = self.select_best_strategy(closest_prey, closest_prey_dist, other_hunters, available_strategies)
            
            # Execute the selected strategy
            if selected_strategy == "formation_hunting":
                best_move = self.execute_formation_hunting(closest_prey.pos, other_hunters)
                logger.info(f"CooperativeHunter {self.unique_id}: Using FORMATION HUNTING strategy")
            elif selected_strategy == "flanking":
                hunters_targeting_same_prey = [h for h in other_hunters 
                                             if h.find_closest_prey()[0] == closest_prey]
                best_move = self.get_flanking_position(closest_prey.pos, hunters_targeting_same_prey)
                logger.info(f"CooperativeHunter {self.unique_id}: Using FLANKING strategy with {len(hunters_targeting_same_prey)} other hunters")
            else:  # direct_pursuit
                best_move = self.find_best_move_towards(closest_prey.pos, closest_prey_dist)
                logger.info(f"CooperativeHunter {self.unique_id}: Using DIRECT PURSUIT strategy (reason: {selected_strategy})")
            
            self.model.grid.move_agent(self, best_move)
        elif self.shared_prey_locations:
            # Move towards shared prey location
            closest_shared_pos = min(self.shared_prey_locations, 
                                   key=lambda pos: self.manhattan_distance(self.pos, pos))
            best_move = self.find_best_move_towards(closest_shared_pos, 
                                                   self.manhattan_distance(self.pos, closest_shared_pos))
            logger.info(f"CooperativeHunter {self.unique_id}: Moving towards shared prey location {closest_shared_pos}")
            self.model.grid.move_agent(self, best_move)
        else:
            # No prey info, move randomly
            logger.info(f"CooperativeHunter {self.unique_id}: No prey info, moving randomly")
            self.random_move()

    def get_flanking_position(self, prey_pos, other_hunters):
        """Calculate best flanking position to surround prey with other hunters."""
        # Assign approach angles based on hunter ID to avoid clustering
        num_hunters = len(other_hunters) + 1  # Including self
        angle_step = 360 / max(num_hunters, 4)  # At least 4 directions
        
        # Sort hunters by ID to ensure consistent angle assignment
        all_hunters = sorted([self] + other_hunters, key=lambda h: h.unique_id)
        my_index = all_hunters.index(self)
        my_angle = my_index * angle_step
        
        logger.info(f"CooperativeHunter {self.unique_id}: Flanking - assigned angle {my_angle:.1f}° "
                   f"(hunter {my_index+1}/{num_hunters}, targeting prey at {prey_pos})")
        
        # Calculate flanking position
        flanking_pos = self.calculate_position_from_angle(prey_pos, my_angle, distance=2)
        
        # Find best move towards flanking position
        current_dist = self.manhattan_distance(self.pos, flanking_pos)
        best_move = self.find_best_move_towards(flanking_pos, current_dist)
        
        logger.info(f"CooperativeHunter {self.unique_id}: Flanking move from {self.pos} to {best_move} "
                   f"(target flanking pos: {flanking_pos})")
        
        return best_move
    
    def calculate_position_from_angle(self, center_pos, angle_degrees, distance):
        """Calculate position at given angle and distance from center."""
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        
        # Calculate offset
        dx = int(round(distance * math.cos(angle_rad)))
        dy = int(round(distance * math.sin(angle_rad)))
        
        # Calculate target position
        target_x = center_pos[0] + dx
        target_y = center_pos[1] + dy
        
        # Ensure position is within grid bounds
        target_x = max(0, min(self.model.grid.width - 1, target_x))
        target_y = max(0, min(self.model.grid.height - 1, target_y))
        
        return (target_x, target_y)
    
    # ===============================
    # INTELLIGENT STRATEGY SELECTION
    # ===============================
    
    def select_best_strategy(self, closest_prey, closest_prey_dist, other_hunters, available_strategies):
        """Intelligent strategy selection based on situation analysis."""
        prey_pos = closest_prey.pos
        
        logger.info(f"CooperativeHunter {self.unique_id}: Strategy analysis - distance={closest_prey_dist}, "
                   f"other_hunters={len(other_hunters)}, prey_pos={prey_pos}")
        
        # Priority 1: Direct pursuit for immediate captures or cornered prey
        if self.should_use_direct_pursuit(prey_pos, closest_prey_dist, other_hunters):
            return "direct_pursuit_immediate"
        
        # Priority 2: Formation hunting for complex situations (only if enabled)
        if ("formation_hunting" in available_strategies and 
            self.should_use_formation_hunting(prey_pos, other_hunters)):
            return "formation_hunting"
        
        # Priority 3: Flanking for coordination opportunities (only if enabled)
        if ("flanking" in available_strategies and 
            self.should_use_flanking(prey_pos, other_hunters)):
            return "flanking"
        
        # Fallback: Direct pursuit
        return "direct_pursuit_fallback"
    
    def should_use_direct_pursuit(self, prey_pos, distance, other_hunters):
        """Determine if direct pursuit is the best strategy."""
        # Use direct pursuit if very close (immediate capture possible)
        if distance <= 1:
            logger.info(f"CooperativeHunter {self.unique_id}: Direct pursuit - immediate capture possible (dist={distance})")
            return True
        
        # Use direct pursuit if prey is cornered
        if self.is_prey_cornered(prey_pos):
            logger.info(f"CooperativeHunter {self.unique_id}: Direct pursuit - prey is cornered at {prey_pos}")
            return True
        
        # Use direct pursuit if I'm the only hunter targeting this prey
        if len(other_hunters) == 0:
            logger.info(f"CooperativeHunter {self.unique_id}: Direct pursuit - solo hunter")
            return True
        
        return False
    
    def should_use_formation_hunting(self, prey_pos, other_hunters):
        """Determine if formation hunting is appropriate."""
        # Need at least 2 other hunters for formation (3 total)
        if len(other_hunters) < 2:
            logger.info(f"CooperativeHunter {self.unique_id}: Formation hunting rejected - need 3+ hunters, have {len(other_hunters)+1}")
            return False
        
        # Check if prey has multiple escape routes (not cornered)
        escape_routes = self.count_prey_escape_routes(prey_pos)
        if escape_routes < 2:
            logger.info(f"CooperativeHunter {self.unique_id}: Formation hunting rejected - prey cornered (escape_routes={escape_routes})")
            return False
        
        # Check if there's enough open space for formation
        open_space = self.calculate_open_space_around(prey_pos, radius=3)
        if open_space < 6:
            logger.info(f"CooperativeHunter {self.unique_id}: Formation hunting rejected - insufficient space (open_space={open_space})")
            return False
        
        logger.info(f"CooperativeHunter {self.unique_id}: Formation hunting approved - hunters={len(other_hunters)+1}, "
                   f"escape_routes={escape_routes}, open_space={open_space}")
        return True
    
    def should_use_flanking(self, prey_pos, other_hunters):
        """Determine if flanking strategy is appropriate."""
        if len(other_hunters) == 0:
            logger.info(f"CooperativeHunter {self.unique_id}: Flanking rejected - no other hunters")
            return False
        
        # Find hunters also targeting this same prey
        hunters_targeting_same_prey = [h for h in other_hunters 
                                     if h.find_closest_prey()[0] and 
                                     h.find_closest_prey()[0].pos == prey_pos]
        
        if len(hunters_targeting_same_prey) == 0:
            logger.info(f"CooperativeHunter {self.unique_id}: Flanking rejected - no other hunters targeting same prey")
            return False
        
        # Check if hunters are clustering from same direction
        my_angle = self.calculate_angle_to_prey(prey_pos)
        clustered_hunters = 0
        for hunter in hunters_targeting_same_prey:
            other_angle = hunter.calculate_angle_to_prey(prey_pos)
            angle_diff = abs(my_angle - other_angle)
            if angle_diff < 90:  # Within 90 degrees = clustered
                clustered_hunters += 1
        
        if clustered_hunters > 0:
            logger.info(f"CooperativeHunter {self.unique_id}: Flanking approved - {clustered_hunters} hunters clustered, "
                       f"my_angle={my_angle:.1f}°")
            return True
        
        # Check if prey has optimal escape routes for flanking (2-3 routes)
        escape_routes = self.count_prey_escape_routes(prey_pos)
        if 2 <= escape_routes <= 3:
            logger.info(f"CooperativeHunter {self.unique_id}: Flanking approved - optimal escape routes ({escape_routes})")
            return True
        
        logger.info(f"CooperativeHunter {self.unique_id}: Flanking rejected - no clustering, escape_routes={escape_routes}")
        return False
    
    # ===============================
    # SITUATIONAL ANALYSIS HELPERS
    # ===============================
    
    def is_prey_cornered(self, prey_pos):
        """Check if prey is cornered (near walls/edges with limited escape routes)."""
        x, y = prey_pos
        width, height = self.model.grid.width, self.model.grid.height
        
        # Count walls adjacent to prey
        walls = 0
        if x == 0: walls += 1  # Left wall
        if x == width - 1: walls += 1  # Right wall
        if y == 0: walls += 1  # Bottom wall
        if y == height - 1: walls += 1  # Top wall
        
        # Prey is cornered if against 2+ walls or has very few escape routes
        escape_routes = self.count_prey_escape_routes(prey_pos)
        is_cornered = walls >= 2 or escape_routes <= 2
        
        logger.info(f"CooperativeHunter {self.unique_id}: Cornered analysis - walls={walls}, escape_routes={escape_routes}, cornered={is_cornered}")
        return is_cornered
    
    def count_prey_escape_routes(self, prey_pos):
        """Count the number of escape routes available to prey."""
        neighbors = self.model.grid.get_neighborhood(prey_pos, moore=True, include_center=False)
        escape_routes = 0
        
        for pos in neighbors:
            # Check if position is within bounds
            if (0 <= pos[0] < self.model.grid.width and 
                0 <= pos[1] < self.model.grid.height):
                # Check if position is free (no agents)
                cell_contents = self.model.grid.get_cell_list_contents([pos])
                if not cell_contents:  # Empty cell = escape route
                    escape_routes += 1
        
        return escape_routes
    
    def calculate_open_space_around(self, prey_pos, radius=3):
        """Calculate available open space around prey position."""
        open_space = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = prey_pos[0] + dx, prey_pos[1] + dy
                if (0 <= x < self.model.grid.width and 
                    0 <= y < self.model.grid.height):
                    cell_contents = self.model.grid.get_cell_list_contents([(x, y)])
                    if not cell_contents:  # Empty cell
                        open_space += 1
        return open_space
    
    def calculate_angle_to_prey(self, prey_pos):
        """Calculate angle from hunter to prey position."""
        dx = prey_pos[0] - self.pos[0]
        dy = prey_pos[1] - self.pos[1]
        
        if dx == 0 and dy == 0:
            return 0  # Same position
        
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360  # Normalize to 0-360 degrees
    
    def execute_formation_hunting(self, prey_pos, other_hunters):
        """Execute formation hunting strategy."""
        # Simple formation: arrange hunters in a pattern around prey
        # Leader (lowest ID) goes direct, others flank from sides
        all_hunters = sorted([self] + other_hunters, key=lambda h: h.unique_id)
        my_index = all_hunters.index(self)
        
        if my_index == 0:
            # Leader - direct approach
            logger.info(f"CooperativeHunter {self.unique_id}: Formation role - LEADER (direct approach)")
            return self.find_best_move_towards(prey_pos, self.manhattan_distance(self.pos, prey_pos))
        else:
            # Flanker - approach from assigned angle
            angle = (my_index * 120) % 360  # Spread around prey
            flanking_pos = self.calculate_position_from_angle(prey_pos, angle, distance=2)
            logger.info(f"CooperativeHunter {self.unique_id}: Formation role - FLANKER {my_index} (angle={angle}°)")
            return self.find_best_move_towards(flanking_pos, self.manhattan_distance(self.pos, flanking_pos))
