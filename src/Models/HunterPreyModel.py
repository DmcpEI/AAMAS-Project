import logging
import numpy as np
from typing import Optional

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from Agents.Hunters.RandomHunter import RandomHunter
from Agents.Hunters.GreedyHunter import GreedyHunter
from Agents.Hunters.NashQHunter import NashQHunter
from Agents.Hunters.MinimaxQHunter import MinimaxQHunter
from Agents.Hunters.CooperativeHunter import CooperativeHunter

from Agents.Preys.Prey import Prey
from Agents.Preys.NashQPrey import NashQPrey
from Agents.Preys.MinimaxQPrey import MinimaxQPrey
from Agents.Preys.CooperativePrey import CooperativePrey

from QTableDisplayer import qtable_displayer

from Models.ModelConfig import ModelConfig

logger = logging.getLogger(__name__)
    

class HunterPreyModel(Model):
    """
    A model with dynamic numbers of hunters and preys on a non-toroidal grid.
    Uses the AgentSet API for scheduling.
    """
    def __init__(
        self,
        N_hunters: int = 10,
        N_greedy_hunters: int = 0,
        N_nash_q_hunters: int = 0,
        N_minimax_q_hunters: int = 0,
        N_preys: int = 50,
        N_nash_q_preys: int = 0,
        N_minimax_q_preys: int = 0,
        N_coop_hunters: int = 0,
        N_coop_preys: int = 0,
        width: int = 20,
        height: int = 20,
        print_step_rewards: bool = True,
        seed: Optional[int] = None,
        # Cooperation Strategy Parameters
        use_formation_hunting: bool = True,
        use_flanking: bool = True,
    ):        # Initialize base Model (sets up self.agents, RNG, etc.)
        super().__init__(seed=seed)
        logger.info(
            f"Initializing model with {N_hunters} hunters, {N_greedy_hunters} greedy hunters, "
            f"{N_nash_q_hunters} nash q-hunters, {N_minimax_q_hunters} minimax q-hunters, "
            f"on a {width}x{height} grid"
        )
        self.num_hunters = N_hunters        
        self.num_greedy_hunters = N_greedy_hunters
        self.num_nash_q_hunters = N_nash_q_hunters
        self.num_minimax_q_hunters = N_minimax_q_hunters
        self.num_preys = N_preys
        self.num_nash_q_preys = N_nash_q_preys
        self.num_minimax_q_preys = N_minimax_q_preys
        self.num_coop_hunters = N_coop_hunters
        self.num_coop_preys = N_coop_preys
        self.print_step_rewards = print_step_rewards

        # Store cooperation strategy settings
        self.use_formation_hunting = use_formation_hunting
        self.use_flanking = use_flanking
        
        self.grid = MultiGrid(width, height, torus=False)        # Kill notification system with persistent display
        self.kill_occurred_this_step = False
        self.kill_info = None  # Will store {'hunter_id': X, 'prey_id': Y, 'hunter_type': 'type', 'position': (x,y)}
        self.kill_notification_duration = ModelConfig.DEFAULT_KILL_NOTIFICATION_DURATION  # Show notification for 1 steps
        self.kill_notification_timer = 0  # Countdown timer for showing notification
        self.pending_hunter_teleports = []  # List of hunters waiting to teleport next step
        self.pending_prey_respawns = []  # List of prey waiting to respawn next step        # Nash Q-Learning synchronization system (3-phase approach)
        self.nash_q_phase = ModelConfig.NASH_Q_PHASE_NORMAL  # "normal", "choose_action", "observe", or "update"
        self.nash_q_experiences = {}  # Store experiences for synchronized learning
          # Hunting tracking to prevent duplicate hunts in same step
        self.hunted_this_step = set()  # Track (hunter_id, prey_id) pairs that already hunted this step        # Reset kill counters
        self._reset_kill_counters()
        
        # Distance-based reward tracking
        self.previous_positions = {}  # Track agent positions from previous step
        self.current_distances = {}   # Track current hunter-prey distances
        self.previous_distances = {}  # Track previous hunter-prey distances
        
        # Initialize coordination system
        self.current_step = 0
        
        # Data collector for live counts
        self.datacollector = DataCollector(self._create_datacollector_config())
        
        # Create and place all agents using configuration-driven approach
        for agent_class, count, agent_name in self._get_agent_configs():
            for _ in range(count):
                agent = agent_class(self)
                
                # Special configuration for cooperative agents
                if isinstance(agent, CooperativeHunter):
                    agent.available_strategies = self.get_available_strategies()
                
                pos = self._get_collision_free_position()
                if pos:
                    self.grid.place_agent(agent, pos)
                else:
                    # Fallback if no collision-free position available
                    x = self.random.randrange(width)
                    y = self.random.randrange(height)
                    self.grid.place_agent(agent, (x, y))
                #logger.debug(f"Placed {agent_name} {agent.unique_id} at {agent.pos}")
       
        # Initial data collection
        self.running = True
        self.datacollector.collect(self)
    def _create_datacollector_config(self):
        """Create the configuration dictionary for DataCollector"""
        return {
            # Agent counts
            "Hunters": lambda m: m.count_type(RandomHunter),
            "GreedyHunters": lambda m: m.count_type(GreedyHunter),
            "NashQHunters": lambda m: m.count_type(NashQHunter),
            "MinimaxQHunters": lambda m: m.count_type(MinimaxQHunter),
            "CoopHunters": lambda m: m.count_type(CooperativeHunter),
            "Preys": lambda m: m.count_type(Prey),
            "NashQPreys": lambda m: m.count_type(NashQPrey),
            "MinimaxQPreys": lambda m: m.count_type(MinimaxQPrey),
            "CoopPreys": lambda m: m.count_type(CooperativePrey),
            
            # Kill statistics
            "RandomHunterKills": lambda m: m.get_agent_kills(RandomHunter),
            "GreedyHunterKills": lambda m: m.get_agent_kills(GreedyHunter),
            "NashQHunterKills": lambda m: m.get_agent_kills(NashQHunter),
            "MinimaxQHunterKills": lambda m: m.get_agent_kills(MinimaxQHunter),
            "CoopHunterKills": lambda m: m.get_agent_kills(CooperativeHunter),
            
            # Reward statistics
            "AvgHunterReward": lambda m: m.avg_agent_reward(RandomHunter),
            "AvgNashQHunterReward": lambda m: m.avg_agent_reward(NashQHunter),
            "AvgMinimaxQHunterReward": lambda m: m.avg_agent_reward(MinimaxQHunter),
            "AvgPreyReward": lambda m: m.avg_agent_reward(Prey),
            "AvgNashQPreyReward": lambda m: m.avg_agent_reward(NashQPrey),
            "AvgMinimaxQPreyReward": lambda m: m.avg_agent_reward(MinimaxQPrey),
        }

    def _get_agent_configs(self):
        """Get agent configuration tuples for initialization"""
        return [
            (RandomHunter, self.num_hunters, "RandomHunter"),
            (GreedyHunter, self.num_greedy_hunters, "GreedyHunter"),
            (NashQHunter, self.num_nash_q_hunters, "NashQHunter"),
            (MinimaxQHunter, self.num_minimax_q_hunters, "MinimaxQHunter"),
            (CooperativeHunter, self.num_coop_hunters, "CooperativeHunter"),
            (Prey, self.num_preys, "Prey"),
            (NashQPrey, self.num_nash_q_preys, "NashQPrey"),
            (MinimaxQPrey, self.num_minimax_q_preys, "MinimaxQPrey"),
            (CooperativePrey, self.num_coop_preys, "CooperativePrey"),
        ]

    def _reset_kill_counters(self):
        """Reset all agent class kill counters to zero"""
        RandomHunter.total_kills = 0
        GreedyHunter.total_kills = 0
        NashQHunter.total_kills = 0
        MinimaxQHunter.total_kills = 0
        CooperativeHunter.total_kills = 0
        
    def _get_collision_free_position(self):
        """Get a random position that doesn't have other agents and is far from hunters."""
        # Get all possible positions
        all_positions = [(x, y) for x in range(self.grid.width) for y in range(self.grid.height)]
         # Get hunter positions
        hunter_positions = [agent.pos for agent in self.agents 
                        if agent.__class__.__name__.endswith("Hunter")]
        
        # Filter positions: empty + at least 2 cells away from any hunter
        safe_positions = []
        for pos in all_positions:
            # Check if cell is empty
            cell_contents = self.grid.get_cell_list_contents([pos])
            if cell_contents:
                continue
                
            # Check distance from hunters (Manhattan distance >= 2)
            safe_from_hunters = True
            for hunter_pos in hunter_positions:
                if hunter_pos:  # Make sure hunter has a position
                    manhattan_dist = abs(pos[0] - hunter_pos[0]) + abs(pos[1] - hunter_pos[1])
                    if manhattan_dist < 2:  # Too close to hunter
                        safe_from_hunters = False
                        break
            
            if safe_from_hunters:
                safe_positions.append(pos)
        
        if safe_positions:
            chosen_pos = self.random.choice(safe_positions)
            logger.info(f"Respawning at safe position {chosen_pos} (distance ≥2 from hunters)")
            return chosen_pos
        else:            # Fallback: just find any empty cell
            for pos in all_positions:
                cell_contents = self.grid.get_cell_list_contents([pos])
                if not cell_contents:
                    logger.warning(f"No safe respawn positions, using emergency position {pos}")
                    return pos
            return None
    
    def _store_previous_positions(self):
        """Store current positions of all agents as previous positions for distance tracking."""
        self.previous_positions = {}
        for agent in self.agents:
            if hasattr(agent, 'pos') and agent.pos is not None:
                self.previous_positions[agent.unique_id] = agent.pos

    def step(self) -> None:
        """Advance the model by one step."""
        logger.debug(f"Model step {self.steps}")

        # Store current positions as previous positions before movement
        self._store_previous_positions()

        # Process pending actions from previous step
        self._process_pending_actions()
        
        # Reset step tracking variables
        self._reset_step_tracking()
        
        # Handle kill notification timer
        self._update_kill_notification_timer()

        # Get all agents in random order and separate by type
        q_learn_agents, other_agents = self._separate_agents_by_type()

        # Execute Nash Q-learning agents with 3-phase synchronization
        if q_learn_agents:
            self._execute_q_learn_agents(q_learn_agents)
            
        # Execute other (non-Nash Q) agents normally
        self._execute_other_agents(other_agents)        # Final checks and cleanup
        self._check_and_perform_hunting()
        self._finalize_step()

    def _process_pending_actions(self):
        """Process pending teleports and respawns from previous kills."""
        print(f"🔧 DEBUG: START _process_pending_actions - Hunters: {len(self.pending_hunter_teleports)}, Preys: {len(self.pending_prey_respawns)}")
        print(f"🔧 DEBUG: Hunter IDs: {[h.unique_id for h in self.pending_hunter_teleports]}")
        print(f"🔧 DEBUG: Prey IDs: {[p.unique_id for p in self.pending_prey_respawns]}")
        try:
            print(f"🔧 DEBUG: Processing pending actions - Hunters: {len(self.pending_hunter_teleports)}, Preys: {len(self.pending_prey_respawns)}")
            # Process pending hunter teleports
            if self.pending_hunter_teleports:
                print(f"🔧 DEBUG: Processing {len(self.pending_hunter_teleports)} hunter teleports")
                for hunter in list(self.pending_hunter_teleports):
                    try:
                        new_pos = self._get_collision_free_position()
                        if new_pos:
                            self.grid.move_agent(hunter, new_pos)
                    except Exception as e:
                        logger.error(f"Error teleporting hunter {hunter.unique_id}: {e}")
                self.pending_hunter_teleports.clear()

            # Process pending prey respawns
            if self.pending_prey_respawns:
                print(f"🔧 DEBUG: Processing {len(self.pending_prey_respawns)} prey respawns")
                for prey in list(self.pending_prey_respawns):
                    try:
                        new_pos = self._get_collision_free_position()
                        if new_pos:
                            self.grid.move_agent(prey, new_pos)
                            if hasattr(prey, '_is_dead'):
                                prey._is_dead = False
                            prey.scheduled_for_removal = False
                            logger.info(f"{prey.__class__.__name__} {prey.unique_id} respawned at {new_pos}")
                    except Exception as e:
                        logger.error(f"Error respawning prey {prey.unique_id}: {e}")                # Clear the list after processing all prey (successful or failed)
                self.pending_prey_respawns.clear()
            else:
                print(f"🔧 DEBUG: No prey respawns to process (list is empty)")
        except Exception as e:
            logger.error(f"Error in _process_pending_actions: {e}")
    
    def _reset_step_tracking(self):
        """Reset tracking variables for the current step."""
        self.kill_occurred_this_step = False
        self.hunted_this_step.clear()
        self.nash_q_experiences.clear()
        # Clear distance reward cache for fresh calculation each step
        if hasattr(self, '_distance_rewards_cache'):
            self._distance_rewards_cache.clear()

    def _update_kill_notification_timer(self):
        """Update the kill notification timer."""
        if self.kill_notification_timer > 0:
            self.kill_notification_timer -= 1
            if self.kill_notification_timer == 0:
                self.kill_info = None

    def _separate_agents_by_type(self):
        """Separate agents into Nash Q-learning agents and others."""
        try:
            agents_list = list(self.agents)
            self.random.shuffle(agents_list)
            
            q_learn_agents = []
            other_agents = []
            for agent in agents_list:
                try:
                    if isinstance(agent, (NashQHunter, NashQPrey, MinimaxQHunter, MinimaxQPrey)):
                        q_learn_agents.append(agent)
                    else:
                        other_agents.append(agent)
                except Exception as e:
                    logger.error(f"Error checking agent type for agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
                    # Default to other_agents if classification fails
                    other_agents.append(agent)
                    
            return q_learn_agents, other_agents
        except Exception as e:
            logger.error(f"Error in _separate_agents_by_type: {e}")
            return [], []    
    def _agents_can_interact(self, agent1, agent2):
        """
        Check if two agents can interact based on Chebyshev distance.
        Agents can interact if they are in the same cell or adjacent cells (including diagonals).
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            bool: True if agents can interact, False otherwise
        """
        try:
            if not hasattr(agent1, 'pos') or not hasattr(agent2, 'pos'):
                logger.warning(f"Agent missing position attribute: {getattr(agent1, 'unique_id', 'unknown')} or {getattr(agent2, 'unique_id', 'unknown')}")
                return False
                
            if agent1.pos is None or agent2.pos is None:
                logger.warning(f"Agent has None position: {getattr(agent1, 'unique_id', 'unknown')} or {getattr(agent2, 'unique_id', 'unknown')}")
                return False
                
            # Calculate Chebyshev distance (includes diagonals)
            chebyshev_distance = max(abs(agent1.pos[0] - agent2.pos[0]), abs(agent1.pos[1] - agent2.pos[1]))
            
            # Agents can interact if they are in same cell or adjacent (Chebyshev distance <= 1)
            can_interact = chebyshev_distance <= ModelConfig.MAX_INTERACTION_DISTANCE
            
            logger.debug(f"Interaction check: Agent {getattr(agent1, 'unique_id', 'unknown')} at {agent1.pos} and Agent {getattr(agent2, 'unique_id', 'unknown')} at {agent2.pos}, distance: {chebyshev_distance}, can_interact: {can_interact}")
            
            return can_interact
            
        except Exception as e:
            logger.error(f"Error checking if agents can interact: {e}")
            return False    
    def _execute_q_learn_agents(self, nash_q_agents):
        """Execute Nash Q-learning agents using 3-phase synchronization with mixed strategies."""
        try:
            # PHASE 1: Simultaneous Action Selection (Mixed Strategies)
            self.nash_q_phase = ModelConfig.NASH_Q_PHASE_CHOOSE_ACTION
            nash_q_data = self._nash_q_phase_1_choose_actions(nash_q_agents)
            
            # PHASE 2: Simultaneous Execution + Observation
            self.nash_q_phase = ModelConfig.NASH_Q_PHASE_OBSERVE
            # Execute all actions simultaneously (includes movement and hunting)
            self._execute_nash_q_actions_with_hunting(nash_q_data)
            # Observe results and calculate rewards
            self._nash_q_phase_2_observe(nash_q_data)
            
            # PHASE 3: Q-Table Updates
            self.nash_q_phase = ModelConfig.NASH_Q_PHASE_UPDATE
            self._update_nash_q_learning_3phase(nash_q_data)
            
            # Update exploration rates for mixed strategies
            self._update_nash_q_exploration_rates(nash_q_agents)
            
        except Exception as e:
            logger.error(f"Error in _execute_nash_q_agents: {e}")
            # Reset phase to normal on error
            self.nash_q_phase = ModelConfig.NASH_Q_PHASE_NORMAL

    def _execute_other_agents(self, other_agents):
        """Execute non-Nash Q agents normally."""
        try:
            for agent in other_agents:
                try:
                    if agent in self.agents:  # Check if agent still exists
                        if hasattr(agent, '_is_dead') and agent._is_dead:
                            logger.debug(f"Skipping step for dead agent {agent.unique_id}")
                            continue
                        agent.step()
                except Exception as e:
                    logger.error(f"Error executing agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
        except Exception as e:
            logger.error(f"Error in _execute_other_agents: {e}")    
    def _finalize_step(self):
        """Final checks and cleanup for the step."""
        try:
            
            # Update current step counter
            self.current_step = self.steps
            
            # Collect metrics
            try:
                self.datacollector.collect(self)        
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
              # Print step rewards for all agents
            self._print_step_rewards()
            
            # Print Q-tables to terminal (if Nash Q agents exist)
            qtable_displayer.create_status_component(self)
        except Exception as e:
            logger.error(f"Error in _finalize_step: {e}")

    def count_type(self, agent_type: type) -> int:
        """Helper to count agents of a given type."""
        count = sum(isinstance(agent, agent_type) for agent in self.agents)
        logger.debug(f"Current {agent_type.__name__} count: {count}")
        return count

    def get_agent_kills(self, agent_class):
        """Generic method to get total kills for agents of the given class"""
        return getattr(agent_class, 'total_kills', 0)    
    # Generic method to get average reward for any agent class
    def avg_agent_reward(self, agent_class):
        """Generic method to get average reward for agents of the given class"""
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, agent_class)]
        return np.mean(rewards) if rewards else 0.0
    def register_kill(self, hunter_agent, prey_agent):
        """Register a kill event"""
        self.kill_occurred_this_step = True
        self.kill_info = {
            'hunter_id': hunter_agent.unique_id,
            'prey_id': prey_agent.unique_id,
            'hunter_type': hunter_agent.__class__.__name__,
            'prey_type': prey_agent.__class__.__name__,
            'position': hunter_agent.pos,
            'step': self.steps
        }
    def _update_nash_q_learning_2phase(self, nash_q_agents_data):
        """Update Q-tables for Nash Q agents using 2-phase synchronized learning."""
        # Separate hunters and preys
        hunters = {agent_id: data for agent_id, data in nash_q_agents_data.items() 
                  if isinstance(data['agent'], NashQHunter)}
        preys = {agent_id: data for agent_id, data in nash_q_agents_data.items() 
                if isinstance(data['agent'], NashQPrey)}
        
        # For each hunter-prey pair that can interact, perform synchronized learning
        for hunter_id, hunter_data in hunters.items():
            hunter = hunter_data['agent']
            hunter_state_before = hunter_data['state_before']
            hunter_action = hunter_data['action']
            hunter_state_after = hunter.get_state()
              # Calculate hunter reward based on current situation
            hunter_reward = ModelConfig.MOVEMENT_COST  # Step penalty
            cell_agents = self.grid.get_cell_list_contents([hunter.pos])
            caught_prey = False
            for cell_agent in cell_agents:
                if cell_agent.__class__.__name__.endswith("Prey"):
                    if getattr(cell_agent, '_is_dead', False):
                        hunter_reward = ModelConfig.SUCCESSFUL_HUNT_REWARD  # Successful hunt
                        caught_prey = True
                        break
                    elif hunter.pos == cell_agent.pos:
                        hunter_reward = ModelConfig.SAME_CELL_REWARD  # Same cell as prey
                        
            # Find interacting prey agents
            for prey_id, prey_data in preys.items():
                prey = prey_data['agent']
                prey_state_before = prey_data['state_before']
                prey_action = prey_data['action']
                prey_state_after = prey.get_state()
                
                # Check if hunter and prey can interact (adjacent or same cell)
                if self._agents_can_interact(hunter, prey):                    # Calculate prey reward
                    prey_reward = ModelConfig.SURVIVAL_REWARD  # Survival reward
                    if caught_prey and prey.pos == hunter.pos:
                        prey_reward = ModelConfig.DEATH_PENALTY  # Death penalty
                    
                    # Update hunter Q-table with joint action
                    if hasattr(hunter, 'update_q_nash'):
                        hunter.update_q_nash(
                            hunter_state_before,
                            hunter_action,
                            hunter_reward,
                            hunter_state_after,
                            prey_action
                        )
                        
                        logger.debug(f"Hunter {hunter.unique_id} Nash Q update: "
                                   f"state={hunter_state_before}, action={hunter_action}, "
                                   f"prey_action={prey_action}, reward={hunter_reward:.3f}")
                    
                    # Update prey Q-table with joint action
                    if hasattr(prey, 'update_q_nash'):
                        prey.update_q_nash(
                            prey_state_before,
                            prey_action,
                            prey_reward,
                            prey_state_after,
                            hunter_action
                        )
                        
                        logger.debug(f"Prey {prey.unique_id} Nash Q update: "
                                   f"state={prey_state_before}, action={prey_action}, "
                                   f"hunter_action={hunter_action}, reward={prey_reward:.3f}")
                    
                    # Set step rewards for visualization
                    hunter._step_reward = hunter_reward
                    prey._step_reward = prey_reward
    def _check_and_perform_hunting(self):
        """Check all hunter positions and perform hunting if hunters and prey are in same cell."""
        try:
            # Get all hunters
            hunters = [agent for agent in self.agents 
                    if agent.__class__.__name__.endswith("Hunter")]
            
            for hunter in hunters:
                try:
                    # Get all agents at hunter's position
                    cell_agents = self.grid.get_cell_list_contents([hunter.pos])
                    
                    # Check for prey in the same cell
                    for agent in cell_agents:
                        try:
                            if agent.__class__.__name__.endswith("Prey"):
                                # Only hunt if prey is not already scheduled for removal and not already hunted this step
                                hunt_pair = (hunter.unique_id, agent.unique_id)
                                
                                if (not getattr(agent, 'scheduled_for_removal', False) and 
                                    not getattr(agent, '_is_dead', False) and 
                                    hunt_pair not in self.hunted_this_step):
                                    
                                    # Perform the hunt
                                    logger.info(f"{hunter.__class__.__name__} {hunter.unique_id} hunted {agent.__class__.__name__} {agent.unique_id} at {hunter.pos}")
                                    print(f"🎯 CATCH! {hunter.__class__.__name__} {hunter.unique_id} caught {agent.__class__.__name__} {agent.unique_id} at cell {hunter.pos}")
                                    
                                    # Track this hunt to prevent duplicates in same step
                                    self.hunted_this_step.add(hunt_pair)
                                    
                                    # Register kill for visualization
                                    self.register_kill(hunter, agent)
                                    
                                    # Set a flag to indicate the prey is scheduled for removal
                                    if hasattr(agent, '_is_dead'):
                                        agent._is_dead = True
                                    else:
                                        agent.scheduled_for_removal = True
                                    
                                    # Schedule prey for respawn in next step
                                    self.pending_prey_respawns.append(agent)
                                    print(f"🔧 DEBUG: Added prey {agent.unique_id} to pending_prey_respawns. List size: {len(self.pending_prey_respawns)}")
                                    
                                    # Schedule hunter teleportation next step
                                    self.pending_hunter_teleports.append(hunter)
                                    print(f"🔧 DEBUG: Added hunter {hunter.unique_id} to pending_hunter_teleports. List size: {len(self.pending_hunter_teleports)}")
                        except Exception as e:
                            logger.error(f"Error processing hunting for hunter {hunter.unique_id} and agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
                except Exception as e:
                    logger.error(f"Error processing hunter {hunter.unique_id} for hunting: {e}")          
        except Exception as e:
            logger.error(f"Error in _check_and_perform_hunting: {e}")
    
    def _calculate_nash_q_reward(self, agent, data):
        """Calculate distance-based reward for Nash Q agent and detect terminal states (kills/deaths)."""
        try:
            # Check for terminal states first (kills/deaths)
            if agent.__class__.__name__.endswith("Hunter"):
                hunt_result = data.get('hunt_result', 0)
                if hunt_result > 0:
                    # Mark this as a terminal state for the hunter (successful kill)
                    data['is_terminal'] = True
                    return hunt_result  # Use actual hunt reward (usually SUCCESSFUL_HUNT_REWARD)
                
                # Double-check for any prey in same cell that might be dead
                try:
                    cell_agents = self.grid.get_cell_list_contents([agent.pos])
                    for cell_agent in cell_agents:
                        if (cell_agent.__class__.__name__.endswith("Prey") and 
                            getattr(cell_agent, '_is_dead', False)):
                            data['is_terminal'] = True
                            return ModelConfig.SUCCESSFUL_HUNT_REWARD  # Successful hunt
                except Exception as e:
                    logger.error(f"Error checking cell contents for hunter {getattr(agent, 'unique_id', 'unknown')}: {e}")
                    
            elif agent.__class__.__name__.endswith("Prey"):
                is_dead = getattr(agent, '_is_dead', False)
                
                # Check for hunters in same cell
                hunters_in_same_cell = []
                try:
                    cell_agents = self.grid.get_cell_list_contents([agent.pos])
                    hunters_in_same_cell = [a for a in cell_agents 
                                          if a.__class__.__name__.endswith("Hunter")]
                except Exception as e:
                    logger.error(f"Error checking cell contents for prey: {e}")
                
                # Check multiple death indicators
                if (is_dead or 
                    getattr(agent, 'scheduled_for_removal', False) or 
                    hunters_in_same_cell):
                    data['is_terminal'] = True
                    return ModelConfig.DEATH_PENALTY  # -10              # Non-terminal states: calculate distance-based rewards
            data['is_terminal'] = False
            return self._calculate_distance_based_reward(agent, data)
            
        except Exception as e:
            logger.error(f"Error in _calculate_nash_q_reward for agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
            # Return safe fallback
            data['is_terminal'] = False
            return 0.0
        
    def _calculate_distance_based_reward(self, agent, data):
        """Calculate distance-based reward for the agent based on distance changes to opponents."""
        try:
            # Get this agent's previous and current positions
            agent_pos_current = agent.pos
            agent_pos_previous = self.previous_positions.get(agent.unique_id)
            
            if agent_pos_previous is None:
                logger.debug(f"No previous position for agent {agent.unique_id}, using movement cost")
                return ModelConfig.MOVEMENT_COST
            
            # Use cached reward if available
            if hasattr(self, '_distance_rewards_cache') and agent.unique_id in self._distance_rewards_cache:
                reward = self._distance_rewards_cache[agent.unique_id]
                logger.debug(f"Using cached distance reward for {agent.__class__.__name__} {agent.unique_id}: {reward}")
                return reward
            
            # Initialize cache if not present
            if not hasattr(self, '_distance_rewards_cache'):
                self._distance_rewards_cache = {}
            
            total_reward = 0.0
            
            # Determine what type of opponents this agent should consider
            if agent.__class__.__name__.endswith("Hunter"):
                # Hunters get rewards based on distance to prey
                opponent_type = "Prey"
            elif agent.__class__.__name__.endswith("Prey"):
                # Prey get rewards based on distance to hunters
                opponent_type = "Hunter"
            else:
                logger.warning(f"Unknown agent type for distance reward: {agent.__class__.__name__}")
                return 0.0
            
            # Calculate distance changes to all opponents
            for other_agent in self.agents:
                if (other_agent.unique_id != agent.unique_id and 
                    other_agent.__class__.__name__.endswith(opponent_type)):
                    
                    try:
                        # Get other agent's previous and current positions
                        other_pos_current = other_agent.pos
                        other_pos_previous = self.previous_positions.get(other_agent.unique_id)
                        
                        # Skip if we don't have previous position for the other agent
                        if other_pos_previous is None:
                            logger.debug(f"No previous position for opponent {other_agent.unique_id}, skipping")
                            continue
                        
                        # Calculate distances before and after movement
                        distance_before = self._calculate_distance(agent_pos_previous, other_pos_previous)
                        distance_after = self._calculate_distance(agent_pos_current, other_pos_current)
                        
                        # Calculate reward based on distance change
                        if distance_after < distance_before:
                            # Distance decreased - hunter benefits, prey suffers
                            if agent.__class__.__name__.endswith("Hunter"):
                                reward = ModelConfig.DISTANCE_DECREASED_REWARD_HUNTER  # +1
                            else:  # Prey
                                reward = ModelConfig.DISTANCE_DECREASED_PENALTY_PREY  # -1
                        elif distance_after > distance_before:
                            # Distance increased - hunter suffers, prey benefits
                            if agent.__class__.__name__.endswith("Hunter"):
                                reward = ModelConfig.DISTANCE_INCREASED_PENALTY_HUNTER  # -1
                            else:  # Prey
                                reward = ModelConfig.DISTANCE_INCREASED_REWARD_PREY  # +1
                        else:
                            # Distance stayed the same
                            reward = 0.0
                        
                        total_reward += reward
                        
                        logger.debug(f"Distance reward for {agent.__class__.__name__} {agent.unique_id}: "
                                   f"dist_before={distance_before:.2f}, dist_after={distance_after:.2f}, "
                                   f"reward={reward} (vs {other_agent.__class__.__name__} {other_agent.unique_id})")
                                   
                    except Exception as e:
                        logger.error(f"Error calculating distance reward between agents "
                                   f"{agent.unique_id} and {other_agent.unique_id}: {e}")
                        continue
            
            """ # Add base movement cost
            total_reward += ModelConfig.MOVEMENT_COST """
            
            # Cache this agent's total reward
            self._distance_rewards_cache[agent.unique_id] = total_reward
                
            logger.debug(f"Final distance reward for {agent.__class__.__name__} {agent.unique_id}: "
                        f"{total_reward}")
                        
            return total_reward
            
        except Exception as e:
            logger.error(f"Error in _calculate_distance_based_reward for agent "
                        f"{getattr(agent, 'unique_id', 'unknown')}: {e}")
            return ModelConfig.MOVEMENT_COST

    def _calculate_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        try:
            if pos1 is None or pos2 is None:
                return float('inf')
            return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
        except Exception as e:
            logger.error(f"Error calculating distance between {pos1} and {pos2}: {e}")
            return float('inf')

    def _get_other_agents_actions(self, agent, nash_q_data):
        """Get actions of ALL other agents for Nash Q-Learning."""
        other_actions = {}
        
        for other_id, other_data in nash_q_data.items():
            if other_id != agent.unique_id:  # Skip self
                other_actions[other_id] = other_data['action']
        
        return other_actions

    def _update_nash_q_learning_3phase(self, nash_q_data):
        """Update Q-tables for all Nash Q agents using synchronized experiences."""
        try:
            logger.debug(f"Nash Q Phase 3: Updating Q-tables for {len(nash_q_data)} agents")
              # Update Q-tables for all Nash Q agents
            for agent_id, data in nash_q_data.items():
                try:
                    agent = data['agent']
                    
                    # Skip if agent no longer exists or is dead
                    if agent not in self.agents or getattr(agent, '_is_dead', False):
                        continue
                    
                    try:
                        state_before = data['state_before']
                        action = data['action']
                        reward = data['reward']
                        state_after = data['state_after']
                        other_actions = data['other_actions']
                    except KeyError as e:
                        logger.error(f"Missing data key for agent {agent_id}: {e}")
                        continue
                    
                    # Check if this is a terminal state and route to appropriate update method
                    is_terminal = data.get('is_terminal', False)
                    
                    # Update Q-table with each interacting agent
                    try:
                        if other_actions:
                            for other_id, other_action in other_actions.items():
                                try:
                                    if is_terminal and hasattr(agent, 'update_q_terminal'):
                                        # Terminal state - use specialized terminal update
                                        agent.update_q_terminal(
                                            state_before,
                                            action,
                                            reward,
                                            other_action
                                        )
                                        
                                        logger.debug(f"Nash Q terminal update: Agent {agent_id} with other {other_id}, "
                                                   f"reward={reward:.3f}, action={action}, other_action={other_action}")
                                    elif hasattr(agent, 'update_q_nash'):
                                        # Regular state - use normal Nash Q update
                                        agent.update_q_nash(
                                            state_before,
                                            action,
                                            reward,
                                            state_after,
                                            other_action
                                        )
                                        
                                        logger.debug(f"Nash Q update: Agent {agent_id} with other {other_id}, "
                                                   f"reward={reward:.3f}, action={action}, other_action={other_action}")
                                except Exception as e:
                                    logger.error(f"Error updating Q-table for agent {agent_id} with other {other_id}: {e}")
                        else:
                            # No interactions - update with None as other action
                            try:
                                if is_terminal and hasattr(agent, 'update_q_terminal'):
                                    # Terminal state - use specialized terminal update
                                    agent.update_q_terminal(
                                        state_before,
                                        action,
                                        reward,
                                        None
                                    )
                                    
                                    logger.debug(f"Nash Q terminal update: Agent {agent_id} (no interactions), "
                                               f"reward={reward:.3f}, action={action}")
                                elif hasattr(agent, 'update_q_nash'):
                                    # Regular state - use normal Nash Q update
                                    agent.update_q_nash(
                                        state_before,
                                        action,
                                        reward,
                                        state_after,
                                        None
                                    )
                                    
                                    logger.debug(f"Nash Q update: Agent {agent_id} (no interactions), "
                                               f"reward={reward:.3f}, action={action}")
                            except Exception as e:
                                logger.error(f"Error updating Q-table for agent {agent_id} (no interactions): {e}")
                    except Exception as e:
                        logger.error(f"Error in Q-table update process for agent {agent_id}: {e}")
                    
                    # Set step reward for visualization
                    try:
                        agent._step_reward = reward
                    except Exception as e:
                        logger.error(f"Error setting step reward for agent {agent_id}: {e}")
                        
                except Exception as e:
                    logger.error(f"Error processing agent {agent_id} in Nash Q learning update: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in _update_nash_q_learning_3phase: {e}")

    def _print_step_rewards(self):
        """Print step rewards for all agents to terminal."""
        # Check if reward printing is enabled
        if not self.print_step_rewards:
            return
            
        # Only print if we have agents with rewards
        agents_with_rewards = [agent for agent in self.agents if hasattr(agent, '_step_reward')]
        
        if not agents_with_rewards:
            return
            
        print(f"\n=== STEP {self.steps} REWARDS ===")
        
        # Group agents by type for cleaner output
        hunters = [a for a in agents_with_rewards if 'Hunter' in a.__class__.__name__]
        preys = [a for a in agents_with_rewards if 'Prey' in a.__class__.__name__]
        
        if hunters:
            print("HUNTERS:")
            for agent in sorted(hunters, key=lambda x: x.unique_id):
                reward = getattr(agent, '_step_reward', 0)
                agent_type = agent.__class__.__name__.replace('Hunter', '')
                print(f"  {agent_type} #{agent.unique_id}: {reward:+.2f}")
        
        if preys:
            print("PREYS:")
            for agent in sorted(preys, key=lambda x: x.unique_id):
                reward = getattr(agent, '_step_reward', 0)
                agent_type = agent.__class__.__name__.replace('Prey', '')
                print(f"  {agent_type} #{agent.unique_id}: {reward:+.2f}")
        
        # Print summary statistics
        total_hunter_reward = sum(getattr(h, '_step_reward', 0) for h in hunters)
        total_prey_reward = sum(getattr(p, '_step_reward', 0) for p in preys)
        print(f"TOTALS - Hunters: {total_hunter_reward:+.2f}, Preys: {total_prey_reward:+.2f}")
        print("=" * 30)

    def _nash_q_phase_1_choose_actions(self, nash_q_agents):
        """Phase 1: All Nash Q agents choose actions simultaneously."""
        try:
            nash_q_data = {}
            for agent in nash_q_agents:
                try:
                    if hasattr(agent, '_is_dead') and agent._is_dead:
                        continue
                        
                    # Store initial state
                    initial_state = agent.get_state()
                    
                    # Agent chooses action based on current state
                    chosen_action = agent.choose_nash_q_action()
                    nash_q_data[agent.unique_id] = {
                        'agent': agent,
                        'state_before': initial_state,
                        'pos_before': agent.pos,
                        'action': chosen_action
                    }
                except Exception as e:
                    logger.error(f"Error in phase 1 for agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
            return nash_q_data
        except Exception as e:
            logger.error(f"Error in _nash_q_phase_1_choose_actions: {e}")
            return {}

    def _execute_nash_q_actions(self, nash_q_data):
        """Execute Nash Q agents' chosen actions (move first, then hunt)."""
        try:
            # STEP 1: Move all agents to their chosen positions
            for agent_id, data in nash_q_data.items():
                try:
                    agent = data['agent']
                    action = data['action']
                    
                    # Skip if agent no longer exists or is dead
                    if agent not in self.agents or getattr(agent, '_is_dead', False):
                        logger.debug(f"Skipping action execution for agent {agent_id} (dead or removed)")
                        continue
                    
                    # Move agent to chosen position
                    if action and action != agent.pos:
                        try:
                            self.grid.move_agent(agent, action)
                        except Exception as e:
                            logger.error(f"Error moving agent {agent_id} to position {action}: {e}")
                            # Continue with other agents
                            
                except Exception as e:
                    logger.error(f"Error processing movement for agent {agent_id}: {e}")
                    continue
                    
            # STEP 2: After all agents have moved, hunters hunt
            for agent_id, data in nash_q_data.items():
                try:
                    agent = data['agent']
                    
                    # Skip if agent no longer exists or is dead
                    if agent not in self.agents or getattr(agent, '_is_dead', False):
                        continue
                    
                    # Only hunters hunt
                    if agent.__class__.__name__.endswith("Hunter"):
                        try:
                            hunt_result = agent.hunt(return_reward=True)  # Get numeric reward
                            # Store immediate hunt result for reward calculation
                            data['hunt_result'] = hunt_result
                        except Exception as e:
                            logger.error(f"Error during hunting for agent {agent_id}: {e}")
                            # Set fallback hunt result
                            data['hunt_result'] = 0.0
                            
                except Exception as e:
                    logger.error(f"Error processing hunting for agent {agent_id}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in _execute_nash_q_actions: {e}")

    def _execute_nash_q_actions_with_hunting(self, nash_q_data):
        """Phase 2: Execute Nash Q agents' actions with integrated hunting.
        
        This method combines movement and hunting in a single, coordinated phase
        to avoid duplicate hunting calls and ensure proper Nash Q workflow.
        """
        try:
            # Use the existing _execute_nash_q_actions method which already handles
            # movement + hunting in proper sequence (move first, then hunt)
            self._execute_nash_q_actions(nash_q_data)
            
        except Exception as e:
            logger.error(f"Error in _execute_nash_q_actions_with_hunting: {e}")
            # Ensure hunt results are set even on error for reward calculation
            for agent_id, data in nash_q_data.items():
                if 'hunt_result' not in data:
                    data['hunt_result'] = 0.0

    def _nash_q_phase_2_observe(self, nash_q_data):
        """Phase 2: All Nash Q agents observe reward, next state, other actions."""
        try:
            for agent_id, data in nash_q_data.items():
                try:
                    agent = data['agent']
                    
                    # Skip if agent no longer exists or is dead
                    if agent not in self.agents or getattr(agent, '_is_dead', False):
                        logger.debug(f"Skipping observation for agent {agent_id} (dead or removed)")
                        continue
                    
                    # Observe next state after all actions executed
                    try:
                        next_state = agent.get_state()
                        data['state_after'] = next_state
                        data['pos_after'] = agent.pos
                    except Exception as e:
                        logger.error(f"Error getting state for agent {agent_id}: {e}")
                        # Use fallback state
                        data['state_after'] = data.get('state_before', None)
                        data['pos_after'] = agent.pos
                    
                    # Calculate reward based on current situation
                    try:
                        reward = self._calculate_nash_q_reward(agent, data)
                        data['reward'] = reward
                    except Exception as e:
                        logger.error(f"Error calculating reward for agent {agent_id}: {e}")
                        # Use fallback reward based on agent type
                        if agent.__class__.__name__.endswith("Hunter"):
                            data['reward'] = ModelConfig.MOVEMENT_COST
                        else:
                            data['reward'] = ModelConfig.SURVIVAL_REWARD
                    
                    # Observe other agents' actions (for interacting agents)
                    try:
                        other_agents_actions = self._get_other_agents_actions(agent, nash_q_data)
                        data['other_actions'] = other_agents_actions
                    except Exception as e:
                        logger.error(f"Error getting other agents' actions for agent {agent_id}: {e}")
                        data['other_actions'] = {}
                        
                except Exception as e:
                    logger.error(f"Error in phase 2 observation for agent {agent_id}: {e}")
                    # Continue with other agents
                    continue
        except Exception as e:
            logger.error(f"Error in _nash_q_phase_2_observe: {e}")    
    def _update_nash_q_exploration_rates(self, nash_q_agents):
        """Update exploration rates for all Nash Q agents."""
        try:
            for agent in nash_q_agents:
                try:
                    # Skip if agent no longer exists or is dead
                    if agent not in self.agents or getattr(agent, '_is_dead', False):
                        continue
                        
                    # Update epsilon (exploration rate)
                    if hasattr(agent, 'update_epsilon'):
                        try:
                            agent.update_epsilon()
                        except Exception as e:
                            logger.error(f"Error updating epsilon for agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
                    
                    # Check for convergence and boost exploration if needed
                    if hasattr(agent, 'reset_exploration_if_converged'):
                        try:
                            agent.reset_exploration_if_converged()
                        except Exception as e:
                            logger.error(f"Error resetting exploration for agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
                            
                except Exception as e:
                    logger.error(f"Error updating exploration rate for agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
        except Exception as e:
            logger.error(f"Error in _update_nash_q_exploration_rates: {e}")

    def get_available_strategies(self):
        """Get available strategies for cooperative hunters based on model settings."""
        strategies = ["direct"]
        if self.use_formation_hunting:
            strategies.append("formation_hunting")
        if self.use_flanking:
            strategies.append("flanking")
        return strategies