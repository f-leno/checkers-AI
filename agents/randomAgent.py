
"""
    Agent that returns random actions for any state
    
    Author: Felipe Leno (f.leno@usp.br)

"""
from agents.agent import Agent
from randomSeeds.randomseeds import Seeds


class RandomAgent(Agent):
    gamma = 0.99
    def __init__(self):
        self.rnd = Seeds().RANDOM_AGENT_SEED
            
    def observe_reward(self,state,action,statePrime,reward):
        pass
    def select_action(self, state):
        return self.rnd.choice(self.environment.get_actions(state))
    
    def set_environment(self,environment,marker):
        """Connects to the domain environment"""
        self.environment = environment
        self.marker = marker 