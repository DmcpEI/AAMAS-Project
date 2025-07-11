"""
Q-Table Displayer Module
=======================

Unified class for handling Q-table display operations in both terminal and frontend.
Manages analysis, formatting, and display of Q-table data for Nash Q-Learning agents.
"""

import logging
import solara
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class QTableDisplayer:
    """Unified class for Q-table analysis, formatting, and display operations."""
    
    def __init__(self):
        """Initialize the Q-table displayer."""
        self.logger = logging.getLogger(__name__)
    
    # =====================================
    # CORE Q-TABLE RETRIEVAL
    # =====================================
    
    def get_agent_q_table(self, model, agent_class_name: str) -> Optional[Dict]:
        """
        Get Q-table from the first agent of specified class.
        
        Args:
            model: Mesa model instance
            agent_class_name: Name of the agent class
            
        Returns:
            Q-table dictionary or None if not found
        """
        for agent in getattr(model, 'agents', []):
            if agent.__class__.__name__ == agent_class_name:
                return getattr(agent, 'Q', None)
        return None
    
    def get_q_table_stats(self, q_table: Dict) -> Dict[str, Any]:
        """
        Get statistics about a Q-table.
        
        Args:
            q_table: Q-table dictionary
            
        Returns:
            Dictionary with statistics
        """
        if not q_table:
            return {"entries": 0, "avg_value": 0, "max_value": 0, "min_value": 0}
        
        values = list(q_table.values())
        return {
            "entries": len(q_table),
            "avg_value": sum(values) / len(values) if values else 0,
            "max_value": max(values) if values else 0,
            "min_value": min(values) if values else 0
        }
    
    # =====================================
    # FORMATTING UTILITIES
    # =====================================
    def pos_to_direction(self, current_pos: Optional[tuple], action_pos: Optional[tuple]) -> str:
        """
        Convert position coordinates to readable direction string.
        
        Args:
            current_pos: Current position as (x, y) tuple
            action_pos: Target position as (x, y) tuple
            
        Returns:
            Human-readable direction string
        """
        if action_pos is None or current_pos is None:
            return "None"
        
        # Handle string cases (like "unknown")
        if isinstance(action_pos, str):
            return action_pos.capitalize()
        
        # Handle cases where action_pos might be other non-tuple type
        if not isinstance(action_pos, (tuple, list)) or not isinstance(current_pos, (tuple, list)):
            return str(action_pos)
        
        try:
            dx = action_pos[0] - current_pos[0]
            dy = action_pos[1] - current_pos[1]
            
            direction_map = {
                (0, 1): "Up",
                (0, -1): "Down", 
                (1, 0): "Right",
                (-1, 0): "Left",
                (1, 1): "Up-Right",
                (1, -1): "Down-Right",
                (-1, 1): "Up-Left",
                (-1, -1): "Down-Left",
                (0, 0): "Stay"  # No movement
            }
            
            return direction_map.get((dx, dy), f"({dx},{dy})")
            
        except (TypeError, IndexError) as e:
            self.logger.warning(f"Error converting position to direction: {e}")
            return str(action_pos)

    def format_state_string(self, state: tuple, agent_type: str = None) -> str:
        """
        Format state tuple for readable display.
        
        Args:
            state: State tuple (pos, others)
            agent_type: Type of agent ("NashQHunter" or "NashQPrey")
            
        Returns:
            Formatted state string
        """
        if len(state) >= 2:
            if agent_type == "NashQHunter":
                return f"[Me={state[0]}, prey={list(state[1])}]"
            elif agent_type == "NashQPrey":
                return f"[Me={state[0]}, hunter={list(state[1])}]"
            else:
                return f"[pos={state[0]}, others={list(state[1])}]"
        return str(state)
    def format_q_entry(self, k: tuple, v: float, agent_type: str = None) -> str:

        """
        Format a Q-table entry for display.
        
        Args:
            k: Q-table key (state, action, other_action)
            v: Q-value

            agent_type: Type of agent ("NashQHunter" or "NashQPrey")

            
        Returns:
            Formatted string representation
        """
        try:
            state, action, other_action = k
            #print(state, action, other_action)

            state_str = self.format_state_string(state, agent_type)
            action_dir = self.pos_to_direction(state[0], action)
            

            other_action_dir = self.pos_to_direction(
                state[1][0] if state[1] else None, 
                other_action
            ) if other_action else "None"
            

            # Customize the output based on agent type
            if agent_type == "NashQHunter":
                return f"State: {state_str}, Action: {action_dir}, PreyAction: {other_action_dir} → Q={v:.3f}"
            elif agent_type == "NashQPrey":
                return f"State: {state_str}, Action: {action_dir}, HunterAction: {other_action_dir} → Q={v:.3f}"
            else:
                return f"State: {state_str}, Action: {action_dir}, OtherAction: {other_action_dir} → Q={v:.3f}"
        except Exception as e:
            self.logger.error(f"Error formatting Q-table entry {k}: {e}")            
            return f"Error formatting entry: {k} → {v:.3f}"
    
    # =====================================
    # TERMINAL DISPLAY METHODS
    # ======================================

    def print_q_table_section(self, model, agent_class_name: str, display_name: str, max_entries: int = 10) -> None:
        """
        Print Q-table section to terminal for a specific agent type.
        
        Args:
            model: Mesa model instance
            agent_class_name: Class name to search for
            display_name: Display name for the agent type
            max_entries: Maximum entries to display
        """
        print(f"\n=== {display_name} Q-table (last {max_entries} entries) ===")
        
        q_table = self.get_agent_q_table(model, agent_class_name)
        #print(q_table)
        if not q_table:
            print(f"No {agent_class_name} found or Q-table empty")
            return
        

        stats = self.get_q_table_stats(q_table)
        print(f"Total entries: {stats['entries']}")
        print(f"Avg Q-value: {stats['avg_value']:.3f}")
        print(f"Max Q-value: {stats['max_value']:.3f}")
        print(f"Min Q-value: {stats['min_value']:.3f}")
        
        print()
            
        count = 0
        for k, v in list(q_table.items())[-max_entries:]:
            try:
                print(f"  {self.format_q_entry(k, v, agent_class_name)}")
                count += 1
            except Exception as e:
                print(f"  Error formatting Q-table entry {k}: {e}")
    def has_nash_q_agents(self, model) -> bool:
        """Check if the model has any Nash Q-learning agents."""
        for agent in getattr(model, 'agents', []):
            if agent.__class__.__name__ in ["NashQHunter", "NashQPrey"]:
                return True
        return False
    
    def has_minimax_q_agents(self, model) -> bool:
        """Check if the model has any Minimax Q-learning agents."""
        for agent in getattr(model, 'agents', []):
            if agent.__class__.__name__ in ["MinimaxQHunter", "MinimaxQPrey"]:
                return True
        return False
    def has_q_learning_agents(self, model) -> bool:
        """Check if the model has any Q-learning agents (Nash Q or Minimax Q)."""
        return self.has_nash_q_agents(model) or self.has_minimax_q_agents(model)
        
    def print_all_q_tables(self, model) -> None:
        """Print all Q-tables to terminal for monitoring (only if Q-learning agents exist)."""
        # Only print if Q-learning agents are present
        if not self.has_q_learning_agents(model):
            return
            
        print("\n" + "="*80)
        print("Q-TABLES ANALYSIS")
        print("="*80)
        
        # Print Nash Q tables if they exist
        if self.has_nash_q_agents(model):
            self.print_q_table_section(model, "NashQHunter", "NASH Q-LEARNING HUNTER")
            self.print_q_table_section(model, "NashQPrey", "NASH Q-LEARNING PREY")
        
        # Print Minimax Q tables if they exist
        if self.has_minimax_q_agents(model):
            self.print_q_table_section(model, "MinimaxQHunter", "MINIMAX Q-LEARNING HUNTER")
            self.print_q_table_section(model, "MinimaxQPrey", "MINIMAX Q-LEARNING PREY")
        
        print("="*80 + "\n")
    
    # =====================================
    # FRONTEND DISPLAY METHODS
    # =====================================

    def generate_q_table_markdown(self, q_table: Optional[Dict], agent_type: str, max_entries: int = 10) -> str:
        """
        Generate markdown representation of Q-table for frontend display.
        
        Args:
            q_table: Q-table dictionary
            agent_type: Type of agent for display
            max_entries: Maximum number of entries to show
            
        Returns:
            Markdown formatted string
        """
        if not q_table:
            return f"## {agent_type} Q-Table\nNo Q-table data available\n\n"
        
        stats = self.get_q_table_stats(q_table)
        markdown = f"""## {agent_type} Q-Table

**Statistics:**
- Total Entries: {stats['entries']}
- Average Q-value: {stats['avg_value']:.3f}
- 🔝 Max Q-value: {stats['max_value']:.3f}
- 🔻 Min Q-value: {stats['min_value']:.3f}

**Sample Entries (last {max_entries} of {stats['entries']}):**
```
"""
            # Extract agent class name from display name
        if "Nash Q-Learning Hunter" in agent_type:
            agent_class_name = "NashQHunter"
        elif "Nash Q-Learning Prey" in agent_type:
            agent_class_name = "NashQPrey"
        elif "Minimax Q-Learning Hunter" in agent_type:
            agent_class_name = "MinimaxQHunter"
        elif "Minimax Q-Learning Prey" in agent_type:
            agent_class_name = "MinimaxQPrey"
        else:
            agent_class_name = "NashQHunter" if "Hunter" in agent_type else "NashQPrey"
          # Show last N entries
        for k, v in list(q_table.items())[-max_entries:]:
            formatted_entry = self.format_q_entry(k, v, agent_class_name)
            markdown += f"{formatted_entry}\n"
        markdown += "```\n\n"
        return markdown
    def create_qtable_view_component(self, model) -> solara.Markdown:
        """
        Create Q-table view component for frontend display.
        
        Args:
            model: Mesa model instance
            
        Returns:
            Solara Markdown component with Q-table information
        """
        try:
            # Check if any Q-learning agents exist - return empty if not
            if not self.has_q_learning_agents(model):
                return solara.Markdown("")
            
            content = ""
            
            # Add Nash Q tables if they exist
            if self.has_nash_q_agents(model):
                hunter_q_table = self.get_agent_q_table(model, "NashQHunter")
                prey_q_table = self.get_agent_q_table(model, "NashQPrey")
                
                hunter_md = self.generate_q_table_markdown(hunter_q_table, "Nash Q-Learning Hunter")
                prey_md = self.generate_q_table_markdown(prey_q_table, "Nash Q-Learning Prey")
                
                content += hunter_md + prey_md
            
            # Add Minimax Q tables if they exist
            if self.has_minimax_q_agents(model):
                minimax_hunter_q_table = self.get_agent_q_table(model, "MinimaxQHunter")
                minimax_prey_q_table = self.get_agent_q_table(model, "MinimaxQPrey")
                
                minimax_hunter_md = self.generate_q_table_markdown(minimax_hunter_q_table, "Minimax Q-Learning Hunter")
                minimax_prey_md = self.generate_q_table_markdown(minimax_prey_q_table, "Minimax Q-Learning Prey")
                
                content += minimax_hunter_md + minimax_prey_md
                    
            return solara.Markdown(content)
        except Exception as e:
            self.logger.error(f"Error creating Q-table view: {e}")
            return solara.Markdown("")

    def create_status_component(self, model) -> None:
        """
        Print Q-tables to terminal (only if Nash Q agents exist).
        This method only handles terminal printing, not status display.
        """
        # Print Q-tables to terminal every step (only if Nash Q agents exist)
        self.print_all_q_tables(model)
    
    # =====================================
    # CONVENIENCE METHODS
    # =====================================

    def get_combined_q_table_display(self, model) -> str:
        """Get combined Q-table display for both agent types."""
        hunter_q_table = self.get_agent_q_table(model, "NashQHunter")
        prey_q_table = self.get_agent_q_table(model, "NashQPrey")
        
        hunter_md = self.generate_q_table_markdown(hunter_q_table, "Nash Q-Learning Hunter")
        prey_md = self.generate_q_table_markdown(prey_q_table, "Nash Q-Learning Prey")
        
        return hunter_md + prey_md


# Create a global instance for easy access
qtable_displayer = QTableDisplayer()
