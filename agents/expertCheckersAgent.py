"""
    Expert agent for Checkers
    Adapted from:This class is a simple adapter to use the original codification
    Author: Felipe Leno (f.leno@usp.br)

"""
import math
from agents.agent import Agent

from randomSeeds.randomseeds import Seeds




class ExpertCheckersAgent(Agent):
    depth = 3
    rnd = None
    originalAgent = None
    
    def __init__(self):
        self.rnd = Seeds().EXPERT_AGENT_SEED
        self.originalAgent = agents.AlphaBetaAgent(self.depth)
        
    def observe_reward(self,state,action,statePrime,reward):
        pass
    def select_action(self, state):
        """
            calling the original Codification
            
        """
        return self.originalAgent.get_action(state)



