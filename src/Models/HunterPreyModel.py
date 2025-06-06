import logging
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from Agents.Hunters.RandomHunter import RandomHunter
from Agents.Hunters.GreedyHunter import GreedyHunter
from Agents.Hunters.NashQHunter import NashQHunter

from Agents.Hunters.MinimaxHunter import MinimaxHunter
from Agents.Preys.Prey import Prey
from Agents.Preys.NashQPrey import NashQPrey

from Agents.Preys.MinimaxPrey import MinimaxPrey

from typing import Optional
import logging
import numpy as np

# Import QTableDisplayer for terminal Q-table printing
from QTableDisplayer import qtable_displayer

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
        N_minimax_hunters: int = 0,
        N_preys: int = 50,
        N_nash_q_preys: int = 0,
        N_minimax_preys: int = 0,
        width: int = 20,
        height: int = 20,
        minimax_search_depth: int = 4,
        print_step_rewards: bool = True,
        seed: Optional[int] = None    ):
        # Initialize base Model (sets up self.agents, RNG, etc.)
        super().__init__(seed=seed)
        logger.info(
            f"Initializing model with {N_hunters} hunters, {N_greedy_hunters} greedy hunters, "
            f"{N_minimax_hunters} minimax hunters, {N_preys} preys, {N_nash_q_preys} Nash Q-learning preys, "
            f"on a {width}x{height} grid"
        )
        self.num_hunters = N_hunters
        self.num_greedy_hunters = N_greedy_hunters
        self.num_nash_q_hunters = N_nash_q_hunters
        self.num_minimax_hunters = N_minimax_hunters
        self.num_preys = N_preys
        self.num_nash_q_preys = N_nash_q_preys
        self.num_minimax_preys = N_minimax_preys
        self.minimax_search_depth = minimax_search_depth
        self.print_step_rewards = print_step_rewards
        self.grid = MultiGrid(width, height, torus=False)        # Kill notification system with persistent display
        self.kill_occurred_this_step = False
        self.kill_info = None  # Will store {'hunter_id': X, 'prey_id': Y, 'hunter_type': 'type', 'position': (x,y)}
        self.kill_notification_duration = 5  # Show notification for 5 steps
        self.kill_notification_timer = 0  # Countdown timer for showing notification
        self.pending_hunter_teleports = []  # List of hunters waiting to teleport next step
        self.pending_prey_respawns = []  # List of prey waiting to respawn next step# Nash Q-Learning synchronization system (2-phase approach)
        self.nash_q_phase = "normal"  # "normal" or "learning"
        self.nash_q_experiences = {}  # Store experiences for synchronized learning
        
        # Hunting tracking to prevent duplicate hunts in same step
        self.hunted_this_step = set()  # Track (hunter_id, prey_id) pairs that already hunted this step
          # Reset kill counters
        RandomHunter.total_kills = 0
        GreedyHunter.total_kills = 0
        NashQHunter.total_kills = 0
        MinimaxHunter.total_kills = 0        
        
        # Data collector for live counts
        self.datacollector = DataCollector({
            "Hunters": lambda m: m.count_type(RandomHunter),
            "GreedyHunters": lambda m: m.count_type(GreedyHunter),
            "NashQHunters": lambda m: m.count_type(NashQHunter),
            "MinimaxHunters": lambda m: m.count_type(MinimaxHunter),
            "Preys": lambda m: m.count_type(Prey),
            "NashQPreys": lambda m: m.count_type(NashQPrey),
            "MinimaxPreys": lambda m: m.count_type(MinimaxPrey),
            "AvgEnergy": self.avg_energy,
            "RandomHunterKills": self.get_random_hunter_kills,
            "GreedyHunterKills": self.get_greedy_hunter_kills,
            "NashQHunterKills": self.get_nash_q_hunter_kills,
            "MinimaxHunterKills": self.get_minimax_hunter_kills,
            "AvgHunterReward": self.avg_hunter_reward,
            "AvgNashQHunterReward": self.avg_nash_q_hunter_reward,
            "AvgMinimaxHunterReward": self.avg_minimax_hunter_reward,
            "AvgPreyReward": self.avg_prey_reward,
            "AvgNashQPreyReward": self.avg_nash_q_prey_reward,
            "AvgMinimaxPreyReward": self.avg_minimax_prey_reward,
        })# Create and place hunter agents
        for _ in range(self.num_hunters):
            hunter = RandomHunter(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(hunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(hunter, (x, y))

            #logger.debug(f"Placed RandomHunter {hunter.unique_id} at {(x, y)}")

        # Create and place greedy hunter agents
        for _ in range(self.num_greedy_hunters):
            ghunter = GreedyHunter(self)

            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(ghunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(ghunter, (x, y))
            #logger.debug(f"Placed GreedyHunter {ghunter.unique_id} at {(x, y)}")        # Create and place Nash Q-learning hunter agents
        for _ in range(self.num_nash_q_hunters):
            nahunter = NashQHunter(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(nahunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(nahunter, (x, y))
            #logger.debug(f"Placed NashQHunter {nahunter.unique_id} at {(x, y)}")

        # Create and place Minimax hunter agents
        for _ in range(self.num_minimax_hunters):
            mihunter = MinimaxHunter(self, search_depth=self.minimax_search_depth)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(mihunter, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(mihunter, (x, y))
            #logger.debug(f"Placed MinimaxHunter {mihunter.unique_id} at {(x, y)}")

        # Create and place prey agents
        for _ in range(self.num_preys):
            prey = Prey(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(prey, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(prey, (x, y))
        for _ in range(self.num_nash_q_preys):
            naprey = NashQPrey(self)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(naprey, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(naprey, (x, y))            #logger.debug(f"Placed NashQPrey {naprey.unique_id} at {(x, y)}")
        
       #instantiate Minimax prey agents
        for _ in range(self.num_minimax_preys):
            miprey = MinimaxPrey(self, search_depth=self.minimax_search_depth)
            pos = self._get_collision_free_position()
            if pos:
                self.grid.place_agent(miprey, pos)
            else:
                # Fallback if no collision-free position available
                x = self.random.randrange(width)
                y = self.random.randrange(height)
                self.grid.place_agent(miprey, (x, y))
            #logger.debug(f"Placed MinimaxPrey {miprey.unique_id} at {(x, y)}")        # Store all agents in a set for easy access
       
        # Initial data collection
        self.running = True
        self.datacollector.collect(self)    
    def _get_collision_free_position(self):
        """Get a random position that doesn't have other agents."""
        # Get all possible positions
        all_positions = [(x, y) for x in range(self.grid.width) for y in range(self.grid.height)]
        # Filter out positions that already have agents
        available_positions = []
        for pos in all_positions:
            cell_contents = self.grid.get_cell_list_contents([pos])
            # Only consider truly empty cells (no agents)
            if not cell_contents:
                available_positions.append(pos)
        if available_positions:
            return self.random.choice(available_positions)
        else:
            # Fallback to any random position if no completely empty cells available
            return None
    
    def step(self) -> None:
        """Advance the model by one step with 3-phase Nash Q-Learning synchronization."""
        logger.debug(f"Model step {self.steps}")

        # Process pending teleports from previous kills
        if self.pending_hunter_teleports:
            for hunter in list(self.pending_hunter_teleports):
                new_pos = self._get_collision_free_position()
                if new_pos:
                    self.grid.move_agent(hunter, new_pos)
            self.pending_hunter_teleports.clear()

        # Process pending prey respawns from previous kills
        if self.pending_prey_respawns:
            for prey in list(self.pending_prey_respawns):
                new_pos = self._get_collision_free_position()
                if new_pos:
                    self.grid.move_agent(prey, new_pos)
                    if hasattr(prey, '_is_dead'):
                        prey._is_dead = False
                    logger.info(f"{prey.__class__.__name__} {prey.unique_id} respawned at {new_pos}")
            self.pending_prey_respawns.clear()        # Reset step tracking
        self.kill_occurred_this_step = False
        self.hunted_this_step.clear()
        self.nash_q_experiences.clear()
        
        # Handle kill notification timer - keep notification visible for multiple steps
        if self.kill_notification_timer > 0:
            self.kill_notification_timer -= 1
            # Keep kill_info visible while timer is active
            if self.kill_notification_timer == 0:
                self.kill_info = None  # Clear notification when timer expires

        # Get all agents in random order for fair execution
        agents_list = list(self.agents)
        self.random.shuffle(agents_list)
        
        # Separate Nash Q agents from other agents
        nash_q_agents = []
        other_agents = []
        for agent in agents_list:
            if isinstance(agent, (NashQHunter, NashQPrey)):
                nash_q_agents.append(agent)
            else:
                other_agents.append(agent)

        # === 3-PHASE NASH Q-LEARNING SYNCHRONIZATION ===
        if nash_q_agents:
            # PHASE 1: Choose Action - All Nash Q agents choose actions simultaneously
            self.nash_q_phase = "choose_action"
            nash_q_data = {}
            for agent in nash_q_agents:
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
                }            # Execute chosen actions (move agents and hunt immediately for hunters)
            for agent_id, data in nash_q_data.items():
                agent = data['agent']
                action = data['action']
                if action and action != agent.pos:
                    self.grid.move_agent(agent, action)
                    
                    # If this is a hunter, hunt immediately after moving
                    if agent.__class__.__name__.endswith("Hunter"):
                        agent.hunt()
            
            # Check for any remaining hunting opportunities (safety check)
            self._check_and_perform_hunting()
            
            # PHASE 2: Observe - All Nash Q agents observe reward, next state, other actions
            self.nash_q_phase = "observe"
            
            for agent_id, data in nash_q_data.items():
                agent = data['agent']
                
                # Observe next state after all actions executed
                next_state = agent.get_state()
                data['state_after'] = next_state
                data['pos_after'] = agent.pos
                
                # Calculate reward based on current situation
                reward = self._calculate_nash_q_reward(agent, data)
                data['reward'] = reward
                
                # Observe other agents' actions (for interacting agents)
                other_agents_actions = self._get_other_agents_actions(agent, nash_q_data)
                data['other_actions'] = other_agents_actions
            
            # Check for hunting after all Nash Q actions
            self._check_and_perform_hunting()
              # PHASE 3: Update Q-table - All Nash Q agents update Q-tables simultaneously
            self.nash_q_phase = "update"
            self._update_nash_q_learning_3phase(nash_q_data)
              # Update exploration rates for all Nash Q agents
            for agent in nash_q_agents:
                if hasattr(agent, 'update_epsilon'):
                    agent.update_epsilon()
                # Check for convergence and boost exploration if needed
                if hasattr(agent, 'reset_exploration_if_converged'):
                    agent.reset_exploration_if_converged()
          # Execute other (non-Nash Q) agents normally
        for agent in other_agents:
            if agent in self.agents:  # Check if agent still exists
                if hasattr(agent, '_is_dead') and agent._is_dead:
                    logger.debug(f"Skipping step for dead agent {agent.unique_id}")
                    continue
                agent.step()
                # Note: Hunters hunt immediately after moving in their step() method

        # Final hunting check (safety check for any missed hunting opportunities)
        self._check_and_perform_hunting()        # Collect metrics
        self.datacollector.collect(self)
        
        # Print Q-tables to terminal once per step (only if Nash Q agents exist)
        qtable_displayer.print_all_q_tables(self)
        
        # Print step rewards for all agents
        self._print_step_rewards()


    def count_type(self, agent_type: type) -> int:
        """Helper to count agents of a given type."""
        count = sum(isinstance(agent, agent_type) for agent in self.agents)
        logger.debug(f"Current {agent_type.__name__} count: {count}")
        return count
    
    def avg_energy(self) -> float:
        """Compute average energy safely as a model method."""
        try:
            agents = [a for a in self.agents if hasattr(a, "energy")]
            return sum(a.energy for a in agents) / len(agents) if agents else 0.0
        except Exception as e:
            logger.warning(f"AvgEnergy failed at step {self.steps}: {e}")
            return 0.0
    
    def get_random_hunter_kills(self):
        return RandomHunter.total_kills

    def get_greedy_hunter_kills(self):

        return GreedyHunter.total_kills
    
    def get_nash_q_hunter_kills(self):
        return NashQHunter.total_kills

    
    def get_minimax_hunter_kills(self):
        return MinimaxHunter.total_kills
    
    def avg_hunter_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, RandomHunter)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_nash_q_hunter_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, NashQHunter)]
        return np.mean(rewards) if rewards else 0.0
     
    def avg_prey_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, Prey)]
        return np.mean(rewards) if rewards else 0.0
    def avg_nash_q_prey_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, NashQPrey)]
        return np.mean(rewards) if rewards else 0.0
        
    def avg_minimax_hunter_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, MinimaxHunter)]
        return np.mean(rewards) if rewards else 0.0
    
    def avg_minimax_prey_reward(self):
        rewards = [getattr(a, '_step_reward', 0) for a in self.agents if isinstance(a, MinimaxPrey)]
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
            hunter_reward = -0.1  # Step penalty
            cell_agents = self.grid.get_cell_list_contents([hunter.pos])
            caught_prey = False
            for cell_agent in cell_agents:
                if cell_agent.__class__.__name__.endswith("Prey"):
                    if getattr(cell_agent, '_is_dead', False):
                        hunter_reward = 10.0  # Successful hunt
                        caught_prey = True
                        break
                    elif hunter.pos == cell_agent.pos:
                        hunter_reward = 5.0  # Same cell as prey
                        
            # Find interacting prey agents
            for prey_id, prey_data in preys.items():
                prey = prey_data['agent']
                prey_state_before = prey_data['state_before']
                prey_action = prey_data['action']
                prey_state_after = prey.get_state()
                
                # Check if hunter and prey can interact (adjacent or same cell)
                if self._agents_can_interact(hunter, prey):
                    # Calculate prey reward
                    prey_reward = 0.1  # Survival reward
                    if caught_prey and prey.pos == hunter.pos:
                        prey_reward = -10.0  # Death penalty
                    
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
        # Get all hunters
        hunters = [agent for agent in self.agents 
                  if agent.__class__.__name__.endswith("Hunter")]
        
        for hunter in hunters:
            # Get all agents at hunter's position
            cell_agents = self.grid.get_cell_list_contents([hunter.pos])
            
            # Check for prey in the same cell
            for agent in cell_agents:
                if agent.__class__.__name__.endswith("Prey"):
                    # Only hunt if prey is not already scheduled for removal and not already hunted this step
                    hunt_pair = (hunter.unique_id, agent.unique_id)
                    
                    if (not getattr(agent, 'scheduled_for_removal', False) and 
                        not getattr(agent, '_is_dead', False) and 
                        hunt_pair not in self.hunted_this_step):
                        
                        # Perform the hunt
                        logger.info(f"{hunter.__class__.__name__} {hunter.unique_id} hunted {agent.__class__.__name__} {agent.unique_id} at {hunter.pos}")
                        
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
                        
                        # Increment kill counter for the hunter
                        hunter.increment_kills()
                        
                        # Schedule hunter teleportation next step
                        self.pending_hunter_teleports.append(hunter)
                        
                        # Continue checking other prey (removed the break to allow multiple hunts)

    def _agents_can_interact(self, agent1, agent2):
        """Check if two agents can observe/interact with each other."""
        pos1 = agent1.pos
        pos2 = agent2.pos        # Same cell or adjacent cells (Manhattan distance <= 1)
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1
    def _calculate_nash_q_reward(self, agent, data):
        """Calculate simple zero-sum reward for Nash Q agent."""
        
        if agent.__class__.__name__.endswith("Hunter"):
            # Hunter rewards: +10 for catch, -0.1 for no catch
            # Check if hunter caught prey this step
            cell_agents = self.grid.get_cell_list_contents([agent.pos])
            for cell_agent in cell_agents:
                if (cell_agent.__class__.__name__.endswith("Prey") and 
                    getattr(cell_agent, '_is_dead', False)):
                    return 10.0  # Successful hunt
            
            # No catch this step
            return -0.1  # Movement cost
                        
        elif agent.__class__.__name__.endswith("Prey"):
            # Prey rewards: -10 if caught, +0.1 if survived
            if getattr(agent, '_is_dead', False):
                return -10.0  # Death penalty
            else:
                return 0.1   # Survival reward
        
        return 0.0

    def _get_other_agents_actions(self, agent, nash_q_data):
        """Get actions of other interacting agents for Nash Q-Learning."""
        other_actions = {}
        
        for other_id, other_data in nash_q_data.items():
            if other_id == agent.unique_id:
                continue  # Skip self
                
            other_agent = other_data['agent']
            
            # Check if agents can interact (same or adjacent cells)
            if self._agents_can_interact(agent, other_agent):
                other_actions[other_id] = other_data['action']
        
        return other_actions

    def _update_nash_q_learning_3phase(self, nash_q_data):
        """Update Q-tables for all Nash Q agents using synchronized experiences."""
        logger.debug(f"Nash Q Phase 3: Updating Q-tables for {len(nash_q_data)} agents")
        
        # Update Q-tables for all Nash Q agents
        for agent_id, data in nash_q_data.items():
            agent = data['agent']
            
            # Skip if agent no longer exists or is dead
            if agent not in self.agents or getattr(agent, '_is_dead', False):
                continue
                
            state_before = data['state_before']
            action = data['action']
            reward = data['reward']
            state_after = data['state_after']
            other_actions = data['other_actions']
            
            # Update Q-table with each interacting agent
            if other_actions:
                for other_id, other_action in other_actions.items():
                    if hasattr(agent, 'update_q_nash'):
                        agent.update_q_nash(
                            state_before,
                            action,
                            reward,
                            state_after,
                            other_action
                        )
                        
                        logger.debug(f"Nash Q update: Agent {agent_id} with other {other_id}, "
                                   f"reward={reward:.3f}, action={action}, other_action={other_action}")
            else:
                # No interactions - update with None as other action
                if hasattr(agent, 'update_q_nash'):
                    agent.update_q_nash(
                        state_before,
                        action,
                        reward,
                        state_after,
                        None
                    )
                    
                    logger.debug(f"Nash Q update: Agent {agent_id} (no interactions), "
                               f"reward={reward:.3f}, action={action}")
            
            # Set step reward for visualization
            agent._step_reward = reward

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