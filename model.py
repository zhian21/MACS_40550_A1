import mesa
import numpy as np
from mesa.datacollection import DataCollector


class SchellingAgent(mesa.Agent):
    """
    represents an agent in a Schelling segregation model simulation. This agent
    has behaviors influenced by tolerance levels and the presence of social influencers.

    Attributes:
        unique_id (int): a unique identifier assigned to the agent.
        model (Model): the instance of the model to which the agent belongs.
        agent_type (int): the type of the agent, used to determine the agent's group.
        pos (tuple): the position of the agent on the grid.
        is_influencer (bool): a flag indicating whether the agent is a social influencer.
        influence_type (str or None): the type of influence ('positive' or 'negative') if the agent is an influencer.
        tolerance_rate (int): the number of similar or majority agents needed around the influencer to be happy.
        radius (int): the radius in which the agent considers its neighbors for the Schelling model's rules.
        majority_agent_type (int): the agent type that is considered the majority in the environment.
        steps_since_last_move (int): counts the steps since the agent last moved.

    Methods:
        reset: reinitializes the agent with its initial parameters.
        step: activates the agent's behavior for a step in the simulation.
        move_based_on_tolerance: moves the agent to an empty space if the tolerance condition is not met.
        move_based_on_influence_and_tolerance: decides the agent's movement based on influence and tolerance.
        double_move: attempts to move the agent twice if under negative influence.
        select_new_position: Rrndomly selects a new position for the agent to move to from a list of possible positions.
    """

    def __init__(self, unique_id, model, agent_type, pos, is_influencer=False, influence_type=None, tolerance_rate=0):
        super().__init__(unique_id, model)
        self.pos = pos
        self.agent_type = agent_type
        self.is_influencer = is_influencer
        self.influence_type = influence_type
        self.tolerance_rate = tolerance_rate
        self.radius=1
        self.majority_agent_type = 0
        self.steps_since_last_move = 0
        self.agents = []

        self.init_params = {
            "unique_id": unique_id,
            "model": model,
            "agent_type": agent_type,
            "pos": pos,
            "is_influencer": is_influencer,
            "influence_type": influence_type,
            "tolerance_rate": tolerance_rate,
        }

    def reset(self):
        self.__init__(**self.init_params)

    def step(self):
        if self.is_influencer:
            self.move_based_on_tolerance()  
        else:
            self.move_based_on_influence_and_tolerance()

    def move_based_on_tolerance(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        
        similar_count = sum(1 for neighbor in neighbors if neighbor.agent_type == self.agent_type)
        majority_count = sum(1 for neighbor in neighbors if neighbor.agent_type == self.model.majority_agent_type)
        
        if similar_count + majority_count < self.tolerance_rate:
            self.model.grid.move_to_empty(self)
 
   
    def move_based_on_influence_and_tolerance(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False)
        influencer_neighbors = [
            agent for agent in neighbors]

        similar = sum(
            1 for neighbor in neighbors if not neighbor.is_influencer and neighbor.agent_type == self.agent_type)

        if similar >= self.model.homophily:
            self.model.happy += 1
        else:
            positive_influence = any(
                agent.influence_type == 'positive' for agent in influencer_neighbors)
            negative_influence = any(
                agent.influence_type == 'negative' for agent in influencer_neighbors)

            if positive_influence:
                pass
            elif negative_influence:
                self.double_move()
            else:
                self.model.grid.move_to_empty(self)

    def double_move(self):
        for _ in range(2):  
            possible_positions = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False, radius=2)
            new_position = self.select_new_position(possible_positions)
            if self.model.grid.is_cell_empty(new_position):
                self.model.grid.move_agent(self, new_position)

    def select_new_position(self, possible_positions):
        if possible_positions:
            index = np.random.randint(0, len(possible_positions))
            return possible_positions[index]
        return None


class Schelling(mesa.Model):
    """
    This class represents a modified version of the Schelling segregation model,
    extended to include the concept of social influencers. 

    Attributes:
        height (int): the vertical size of the grid.
        width (int): the horizontal size of the grid.
        density (float): the proportion of the grid to be occupied by agents.
        minority_pc (float): the percentage of agents that are of the minority type.
        homophily (int): the number of similar agents that make an agent happy.
        num_type1 (int): the number of type 1(positive) influencers to be placed in the grid.
        tolerance_rate_type1 (int): the tolerance rate for type 1 influencers.
        num_type2 (int): the number of type 2(negative) influencers to be placed in the grid.
        tolerance_rate_type2 (int): The tolerance rate for type 2 influencers.
        schedule (mesa.time.RandomActivation): the scheduler to activate agents each step.
        grid (mesa.space.SingleGrid): the grid where agents are placed.
        happy (int): a count of agents that are currently happy with their location.
        majority_agent_type (int): Tte agent type that is considered the majority (not actively used).
        datacollector (mesa.DataCollector): collects and stores data on the model during simulation.

    Methods:
        populate_agents: initializes the grid by randomly placing agents and influencers.
        step: advances the model by one step, activating each agent's step method in random order.
    """
    def __init__(self, height=20, width=20, density=0.8, minority_pc=0.2, homophily=3,
                 num_type1=1, tolerance_rate_type1=8,
                 num_type2=1, tolerance_rate_type2=2):
        super().__init__()
        self.height = height
        self.width = width
        self.density = density
        self.minority_pc = minority_pc
        self.homophily = homophily

        # parameters for social influencers
        self.num_type1 = num_type1
        self.tolerance_rate_type1 = tolerance_rate_type1
        self.num_type2 = num_type2
        self.tolerance_rate_type2 = tolerance_rate_type2

        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.SingleGrid(width, height, torus=False)
        self.happy = 0
        self.majority_agent_type = 0

        self.datacollector = DataCollector(
            model_reporters={"Happy": lambda m: m.happy} 
        )
        
        self.populate_agents()

    def populate_agents(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.random.random() < self.density:
                    agent_type = 0 if np.random.random() > self.minority_pc else 1
                    is_influencer = False
                    influence_type = None
                    tolerance_rate = None
                    
                    if self.num_type1 > 0:
                        is_influencer = True
                        influence_type = 'positive'
                        tolerance_rate = self.tolerance_rate_type1
                        self.num_type1 -= 1
                    elif self.num_type2 > 0:
                        is_influencer = True
                        influence_type = 'negative'
                        tolerance_rate = self.tolerance_rate_type2
                        self.num_type2 -= 1

                    agent = SchellingAgent(self.next_id(), self, agent_type, (i, j), is_influencer, influence_type, tolerance_rate)
                    self.grid.place_agent(agent, (i, j))
                    self.schedule.add(agent)

    def step(self):
        self.happy = 0
        self.schedule.step()