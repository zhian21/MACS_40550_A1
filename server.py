import mesa
from model import Schelling

def get_happy_agents(model):
    """
    Display a text count of how many happy agents there are.
    """
    return f"Happy agents: {model.happy}"

def schelling_draw(agent):
    """
    Portrayal Method for canvas. Differentiates between normal agents and social influencers.
    """
    if agent is None:
        return
    portrayal = {"Shape": "circle", "r": 0.5, "Filled": "true", "Layer": 0}

    if agent.is_influencer:
        portrayal["Shape"] = "rect"  
        portrayal["w"] = 0.8
        portrayal["h"] = 0.8
        portrayal["Color"] = "#FFA500" if agent.influence_type == 'positive' else "#FF0000"
        portrayal["stroke_color"] = "#000000"
    else:
        portrayal["Color"] = "#808080" if agent.agent_type == 0 else "#0000FF"
        portrayal["stroke_color"] = "#FFFFFF"

    return portrayal

canvas_element = mesa.visualization.CanvasGrid(schelling_draw, 20, 20, 500, 500)

model_params = {
    "height": 20,
    "width": 20,
    "density": mesa.visualization.Slider(
        name="Agent Density", value=0.3, min_value=0.1, max_value=1.0, step=0.05
    ),
    "minority_pc": mesa.visualization.Slider(
        name="Minority Percentage", value=0.2, min_value=0.0, max_value=0.5, step=0.05
    ),
    "homophily": mesa.visualization.Slider(
        name="Homophily", value=3, min_value=0, max_value=8, step=1
    ),
    "num_type1": mesa.visualization.Slider(
        name="Number of Type 1 Influencers", value=1, min_value=0, max_value=10, step=1
    ),
    "tolerance_rate_type1": mesa.visualization.Slider(
        name="Tolerance Rate Type 1", value=1, min_value=0, max_value=10, step=1
    ),
    "num_type2": mesa.visualization.Slider(
        name="Number of Type 2 Influencers", value=1, min_value=0, max_value=10, step=1
    ),
    "tolerance_rate_type2": mesa.visualization.Slider(
        name="Tolerance Rate Type 2", value=6, min_value=0, max_value=8, step=1
    ),
}


server = mesa.visualization.ModularServer(
    Schelling,
    [canvas_element, get_happy_agents],
    "Schelling Segregation Model with Social Influencers",
    model_params,
)
