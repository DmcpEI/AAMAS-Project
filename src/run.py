"""
Hunter-Prey Simulation with Nash Q-Learning
==========================================

A Mesa-based multi-agent simulation featuring hunters and prey agents
with Nash Q-Learning capabilities. Provides real-time visualization
and Q-table monitoring through Solara interface.

Author: AASMA Project Team
"""

import logging
from typing import Dict, Any, Optional, List
import solara
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from Agents.Hunters.RandomHunter import RandomHunter
from Agents.Hunters.GreedyHunter import GreedyHunter
from Agents.Hunters.NashQHunter import NashQHunter
from Agents.Hunters.MinimaxHunter import MinimaxHunter
from Agents.Preys.NashQPrey import NashQPrey
from Agents.Preys.MinimaxPrey import MinimaxPrey
from Agents.Preys.Prey import Prey
from Models.HunterPreyModel import HunterPreyModel
from QTableDisplayer import qtable_displayer
from ChartMetrics import chart_metrics

# Configuration
DEFAULT_SIMULATION_PARAMS = {
    "N_hunters": {"type": "SliderInt", "value": 0, "min": 0, "max": 10, "step": 1, "label": "Number of Hunters"},
    "N_preys": {"type": "SliderInt", "value": 0, "min": 0, "max": 20, "step": 1, "label": "Number of Preys"},
    "N_greedy_hunters": {"type": "SliderInt", "value": 0, "min": 0, "max": 10, "step": 1, "label": "Number of Greedy Hunters"},
    "N_nash_q_hunters": {"type": "SliderInt", "value": 0, "min": 0, "max": 10, "step": 1, "label": "Number of Nash Q-learning Hunters"},
    "N_nash_q_preys": {"type": "SliderInt", "value": 0, "min": 0, "max": 20, "step": 1, "label": "Number of Nash Q-learning Preys"},    
    "N_minimax_hunters": {"type": "SliderInt", "value": 1, "min": 0, "max": 10, "step": 1, "label": "Number of Minimax Hunters"},
    "N_minimax_preys": {"type": "SliderInt", "value": 1, "min": 0, "max": 20, "step": 1, "label": "Number of Minimax Preys"},
    "minimax_search_depth": {"type": "SliderInt", "value": 4, "min": 1, "max": 6, "step": 1, "label": "Minimax Search Depth"},
    "width": {"type": "SliderInt", "value": 4, "min": 1, "max": 20, "step": 1, "label": "Grid Width"},
    "height": {"type": "SliderInt", "value": 4, "min": 1, "max": 20, "step": 1, "label": "Grid Height"},
}

def setup_logging() -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("MESA.mesa.model").setLevel(logging.WARNING)
    return logger

logger = setup_logging()

# Agent visual configuration and portrayal
def get_agent_portrayal_config() -> Dict[str, Dict[str, Any]]:
    """Get agent visual configuration mapping."""
    return {
        RandomHunter: {"color": "#E7E43C", "shape": "circle", "r": 0.5},
        GreedyHunter: {"color": "#FFE66D", "shape": "circle", "r": 0.5},
        NashQHunter: {"color": "#FF5733", "shape": "circle", "r": 0.8},
        MinimaxHunter: {"color": "#8B0000", "shape": "square", "r": 0.8},
        NashQPrey: {"color": "#33FF57", "shape": "circle", "r": 0.8},
        MinimaxPrey: {"color": "#008000", "shape": "square", "r": 0.8},
        Prey: {"color": "#006EB8", "shape": "circle", "r": 0.5},
    }

def agent_portrayal(agent) -> Dict[str, Any]:
    """
    Define visual properties for different agent types.
    
    Args:
        agent: Mesa agent instance
        
    Returns:
        Dictionary containing visual properties (color, shape, size)
    """
    portrayal_config = get_agent_portrayal_config()
    
    # Check for exact type matches first
    for agent_type, config in portrayal_config.items():
        if isinstance(agent, agent_type):
            portrayal = config.copy()
            return portrayal
    
    # Default fallback for unknown agent types
    return {"color": "#CCCCCC", "shape": "circle", "r": 0.3}

# Solara Components (using QTableDisplayer)
def QTableView(model) -> solara.Markdown:
    """Component to display Q-tables for Nash Q-Learning agents."""
    return qtable_displayer.create_qtable_view_component(model)

def has_minimax_agents(model) -> bool:
    """Check if the model has any Minimax agents."""
    for agent in getattr(model, 'agents', []):
        if agent.__class__.__name__ in ["MinimaxHunter", "MinimaxPrey"]:
            return True
    return False

def StatusText(model) -> solara.Markdown:
    """Component to display current simulation status and metrics."""
    try:
        # Trigger Q-table printing to terminal (if Nash Q agents exist)
        qtable_displayer.create_status_component(model)
        
        # Get the last collected data
        data = model.datacollector.get_model_vars_dataframe().iloc[-1]        
        
        # Check what types of agents we have
        has_nash_q = qtable_displayer.has_nash_q_agents(model)
        has_minimax = has_minimax_agents(model)
        
        status_lines = []
        
        if has_nash_q or has_minimax:
            # Map metrics to display names, filtered by agent type
            nash_q_metrics = {
                "NashQHunters": "NashQ Hunters",
                "NashQPreys": "NashQ Preys",
                "NashQHunterKills": "NashQ Hunter Kills",
                "AvgNashQHunterReward": "Avg NashQ Hunter Reward",
                "AvgNashQPreyReward": "Avg NashQ Prey Reward"
            }
            
            minimax_metrics = {
                "MinimaxHunters": "Minimax Hunters",
                "MinimaxPreys": "Minimax Preys",
                "MinimaxHunterKills": "Minimax Hunter Kills",
                "AvgMinimaxHunterReward": "Avg Minimax Hunter Reward",
                "AvgMinimaxPreyReward": "Avg Minimax Prey Reward"
            }
            
            # Only show Nash Q metrics if Nash Q agents exist
            if has_nash_q:
                for metric, display_name in nash_q_metrics.items():
                    if metric in data:
                        if "Avg" in metric and "Reward" in metric:
                            status_lines.append(f"- {display_name}: {data[metric]:.2f}")
                        else:
                            status_lines.append(f"- {display_name}: {int(data[metric])}")
            
            # Only show Minimax metrics if Minimax agents exist
            if has_minimax:
                for metric, display_name in minimax_metrics.items():
                    if metric in data:
                        if "Avg" in metric and "Reward" in metric:
                            status_lines.append(f"- {display_name}: {data[metric]:.2f}")
                        else:
                            status_lines.append(f"- {display_name}: {int(data[metric])}")
            
            additional_info = ""
            if has_nash_q:
                additional_info += "*Check console for detailed Nash Q-table analysis.*"
            if has_minimax:
                if additional_info:
                    additional_info += "\n"
                additional_info += "*Minimax agents use tree search with alpha-beta pruning.*"
            
            status_text = f"""## Simulation Status

**Step {model.steps}:**

{chr(10).join(status_lines)}

{additional_info}
            """
        else:
            # Show basic simulation info without advanced agent specific metrics
            status_text = f"""## Simulation Status

**Step {model.steps}:**

- Total Hunters: {len([a for a in model.agents if 'Hunter' in a.__class__.__name__])}
- Total Prey: {len([a for a in model.agents if 'Prey' in a.__class__.__name__])}
- Total Agents: {len(model.agents)}
            """
        
        return solara.Markdown(status_text)
    except Exception as e:
        logger.error(f"Error generating status text: {e}")
        return solara.Markdown(f"**Error**: Unable to display status - {e}")

def KillNotification(model) -> solara.VBox:
    """Component to display kill notifications as popup-like alerts."""
    kill_info = model.kill_info
    
    if kill_info is None or not model.kill_occurred_this_step:
        # No kill this step, return empty container
        return solara.VBox([])
    
    # Create notification message
    hunter_type = kill_info['hunter_type']
    prey_type = kill_info['prey_type']
    position = kill_info['position']
    step = kill_info['step']
    
    # Remove the "Hunter" or "Prey" suffix for cleaner display
    hunter_display = hunter_type.replace('Hunter', '')
    prey_display = prey_type.replace('Prey', '')
    
    notification_style = {
        'background-color': '#ff4444',
        'color': 'white',
        'padding': '15px',
        'border-radius': '8px',
        'margin': '10px',
        'font-weight': 'bold',
        'text-align': 'center',
        'border': '3px solid #cc0000',
        'box-shadow': '0 4px 8px rgba(0,0,0,0.3)'
    }
    
    notification_text = f"""
## 🎯 KILL ALERT! 🎯

**{hunter_display} Hunter #{kill_info['hunter_id']}** caught **{prey_display} Prey #{kill_info['prey_id']}**

📍 **Position:** {position}  
⏰ **Step:** {step}
    """.strip()
    
    return solara.VBox([
        solara.Markdown(notification_text, style=notification_style)
    ])

# Main simulation initialization
def create_visualization_components():
    """
    Create and configure visualization components.
    
    Returns:
        Tuple of (SpaceGrid, PopChart) components
    """
    SpaceGrid = make_space_component(agent_portrayal)
    PopChart = make_plot_component(chart_metrics.metrics)
    return SpaceGrid, PopChart

def create_model_from_params(model_params: Dict[str, Dict[str, Any]]) -> HunterPreyModel:
    """
    Create model instance from parameter configuration.
    
    Args:
        model_params: Model parameter configuration
        
    Returns:
        Configured HunterPreyModel instance
    """
    param_values = {k: v["value"] for k, v in model_params.items() if isinstance(v, dict)}
    return HunterPreyModel(**param_values)

def create_simulation_page() -> SolaraViz:
    """
    Create the main simulation page with all components.
    
    Returns:
        Configured SolaraViz page
    """    # Create visualization components
    SpaceGrid, PopChart = create_visualization_components()
    
    # Create model instance
    model = create_model_from_params(DEFAULT_SIMULATION_PARAMS)
    
    # Create and configure the page
    page = SolaraViz(
        model,
        components=[KillNotification, SpaceGrid, StatusText, PopChart, QTableView],
        model_params=DEFAULT_SIMULATION_PARAMS,
        name="Hunter-Prey Nash Q-Learning Simulation"
    )
    
    return page

# Main execution
if __name__ == "__main__":
    page = create_simulation_page()
    logger.info("Hunter-Prey simulation initialized successfully")
else:
    # For module import
    page = create_simulation_page()