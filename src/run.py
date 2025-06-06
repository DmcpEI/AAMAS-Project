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
from Agents.Hunters.MinimaxQHunter import MinimaxQHunter
from Agents.Preys.NashQPrey import NashQPrey
from Agents.Preys.MinimaxQPrey import MinimaxQPrey

from Agents.Preys.Prey import Prey
from Models.HunterPreyModel import HunterPreyModel
from QTableDisplayer import qtable_displayer
from ChartMetrics import chart_metrics

# Configuration
DEFAULT_SIMULATION_PARAMS = {
    "N_hunters": {"type": "SliderInt", "value": 0, "min": 0, "max": 10, "step": 1, "label": "Number of Hunters"},
    "N_preys": {"type": "SliderInt", "value": 0, "min": 0, "max": 20, "step": 1, "label": "Number of Preys"},
    "N_greedy_hunters": {"type": "SliderInt", "value": 0, "min": 0, "max": 10, "step": 1, "label": "Number of Greedy Hunters"},
    "N_nash_q_hunters": {"type": "SliderInt", "value": 1, "min": 0, "max": 10, "step": 1, "label": "Number of Nash Q-learning Hunters"},
    "N_nash_q_preys": {"type": "SliderInt", "value": 1, "min": 0, "max": 20, "step": 1, "label": "Number of Nash Q-learning Preys"},
    "N_minimax_q_hunters": {"type": "SliderInt", "value": 0, "min": 0, "max": 10, "step": 1, "label": "Number of Minimax Q-learning Hunters"},
    "N_minimax_q_preys": {"type": "SliderInt", "value": 0, "min": 0, "max": 20, "step": 1, "label": "Number of Minimax Q-learning Preys"},
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
        MinimaxQHunter: {"color": "#DC143C", "shape": "rect", "r": 0.8},
        NashQPrey: {"color": "#33FF57", "shape": "circle", "r": 0.8},
        MinimaxQPrey: {"color": "#228B22", "shape": "rect", "r": 0.8},
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


def has_minimax_q_agents(model) -> bool:
    """Check if the model has any MinimaxQ agents."""
    for agent in getattr(model, 'agents', []):
        if agent.__class__.__name__ in ["MinimaxQHunter", "MinimaxQPrey"]:
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
        has_minimax_q = has_minimax_q_agents(model)
        
        status_lines = []
        
        if has_nash_q or has_minimax_q:
            # Map metrics to display names, filtered by agent type
            nash_q_metrics = {

                "NashQHunters": "NashQ Hunters",
                "NashQPreys": "NashQ Preys",
                "NashQHunterKills": "NashQ Hunter Kills",
                "AvgNashQHunterReward": "Avg NashQ Hunter Reward",
                "AvgNashQPreyReward": "Avg NashQ Prey Reward",            }
            
            minimax_q_metrics = {
                "MinimaxQHunters": "MinimaxQ Hunters",
                "MinimaxQPreys": "MinimaxQ Preys",
                "MinimaxQHunterKills": "MinimaxQ Hunter Kills",
                "AvgMinimaxQHunterReward": "Avg MinimaxQ Hunter Reward",
                "AvgMinimaxQPreyReward": "Avg MinimaxQ Prey Reward"
            }
            
            # Only show Nash Q metrics if Nash Q agents exist
            if has_nash_q:
                for metric, display_name in nash_q_metrics.items():
                    if metric in data:
                        if "Avg" in metric and "Reward" in metric:
                            status_lines.append(f"- {display_name}: {data[metric]:.2f}")
                        else:
                            status_lines.append(f"- {display_name}: {int(data[metric])}")                       
            # Only show MinimaxQ metrics if MinimaxQ agents exist
            if has_minimax_q:
                for metric, display_name in minimax_q_metrics.items():
                    if metric in data:
                        if "Avg" in metric and "Reward" in metric:
                            status_lines.append(f"- {display_name}: {data[metric]:.2f}")
                        else:
                            status_lines.append(f"- {display_name}: {int(data[metric])}")
            
            additional_info = ""
           
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
def DynamicPopChart(model):
    """
    Dynamic population chart component that updates metrics based on agent types.
    
    Args:
        model: Mesa model instance
        
    Returns:
        Solara component with dynamic chart metrics
    """
    try:
        # Update chart metrics based on current model agents
        chart_metrics.update_active_metrics(model)
        
        # Create a new plot component with updated metrics
        current_metrics = chart_metrics.metrics
        
        # Get datacollector data
        if hasattr(model, 'datacollector') and model.datacollector.model_vars:
            data = model.datacollector.get_model_vars_dataframe()
            
            # Filter data to only include active metrics that exist in the data
            available_metrics = [metric for metric in current_metrics if metric in data.columns]
            
            if available_metrics and len(data) > 0:
                # Create plot component with filtered data
                PlotComponent = make_plot_component(available_metrics)
                return PlotComponent(model)
            else:
                return solara.Markdown("**Chart**: No data available yet")
        else:
            return solara.Markdown("**Chart**: Data collector not initialized")
            
    except Exception as e:
        logger.error(f"Error creating dynamic pop chart: {e}")
        return solara.Markdown(f"**Chart Error**: {e}")

def create_visualization_components():
    """
    Create and configure visualization components.
    
    Returns:
        Tuple of (SpaceGrid, DynamicPopChart) components
    """
    SpaceGrid = make_space_component(agent_portrayal)
    return SpaceGrid, DynamicPopChart

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
    """    
    # Create visualization components
    SpaceGrid, DynamicPopChart = create_visualization_components()
    
    # Create model instance
    model = create_model_from_params(DEFAULT_SIMULATION_PARAMS)
    
    # Update chart metrics based on active agents
    chart_metrics.update_active_metrics(model)
      # Create and configure the page
    page = SolaraViz(
        model,
        components=[SpaceGrid, StatusText, DynamicPopChart, KillNotification, QTableView],
        model_params=DEFAULT_SIMULATION_PARAMS,
        name="Hunter-Prey Nash Q-Learning Simulation"
    )
    
    # Add callback to update metrics when model changes
    def on_model_update(new_model):
        chart_metrics.update_active_metrics(new_model)
    
    page.on_model_change = on_model_update
    
    return page

# Main execution
if __name__ == "__main__":
    page = create_simulation_page()
    logger.info("Hunter-Prey simulation initialized successfully")
else:
    # For module import
    page = create_simulation_page()