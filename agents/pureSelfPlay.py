"""
    Implementation of pure self play, that is, the agent always play against the last version of itself
    
    Author: Felipe Leno (f.leno@usp.br)

"""

from agents.qLearning import QLearningAgent
import copy
from randomSeeds.randomseeds import Seeds


class PureSelfPlayAgent(QLearningAgent):
    
    fixedPolicy = False #If true, the agent never explores
    rnd = None
    
    def __init__(self,gamma=0.99, alpha=0.2, marker=None):
        super().__init__(gamma, alpha, marker)
        self.rnd = Seeds().SELFPLAY_AGENT_SEED
    
    def create_fictitious(self,expertSteps, marker=None):
        """
           Returns a fictitious agent with fixed policy
        """
        
        
        agent = copy.deepcopy(self)
        agent.fixedPolicy = True
        
        return agent
    
    """
        The other functions are the same as Q-Learning, but turning off exploration if a copy agent is used
    """
    def observe_reward(self,state,action,statePrime,reward):
        if self.fixedPolicy:
            self.exploring = False
        return super().observe_reward(state, action, statePrime, reward)
    
    
    
    def select_action(self, state):
        if self.fixedPolicy:
            self.exploring = False
        return super().select_action( state)