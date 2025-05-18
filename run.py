import logging
import solara
from solara import reactive
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from mesa.visualization.utils import update_counter
#from mesa.visualization.UserParam import SliderInt  # new namespaced params

from Agents.Hunter import Hunter
from Agents.Prey import Prey
from Models.HunterPreyModel import HunterPreyModel

logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("MESA.mesa.model").setLevel(logging.WARNING)

def agent_portrayal(agent):
    if isinstance(agent, Hunter):
        return {"color": "red",  "r": 0.5}
    else:
        return {"color": "blue", "r": 0.5}

# Define your sliders just like before
model_params = {
    "N_hunters": {"type": "SliderInt", "value": 1, "min": 1, "max": 10, "step": 1, "label": "Number of Hunters"},
    "N_preys":   {"type": "SliderInt", "value": 5, "min": 1, "max": 20, "step": 1, "label": "Number of Preys"},
    "width":     {"type": "SliderInt", "value": 5, "min": 1, "max": 20, "step": 1, "label": "Grid Width"},
    "height":    {"type": "SliderInt", "value": 5, "min": 1, "max": 20, "step": 1, "label": "Grid Height"},
}

# Build Solara components
SpaceGrid = make_space_component(agent_portrayal)  
PopChart  = make_plot_component(["Hunters", "Preys"])  


@solara.component
def StatusText(model):
    update_counter.get() #update in each timestep
    #get the last collected data
    data = model.datacollector.get_model_vars_dataframe().iloc[-1]
    return solara.Markdown(f"Step {model.steps}: "
                          f"Hunters: {int(data['Hunters'])}\n"
                          f"Preys: {int(data['Preys'])}\n"
                          )

# Create the page
page = SolaraViz(
    HunterPreyModel(**{k: v["value"] for k, v in model_params.items() if isinstance(v, dict)}),
    components=[ SpaceGrid, StatusText],
    model_params=model_params,
    name="Hunter-Prey Model"    
)