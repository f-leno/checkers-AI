"""
    Implementation of Q-Learning with Linear Q-function approximation
    
    Author: Felipe Leno (f.leno@usp.br)

"""

import math

from agents.qLearning import QLearningAgent

import numpy as np
from randomSeeds.randomseeds import Seeds
from game import CHECKERS_FEATURE_COUNT
from environment.checkersEnvironment import REWARD_WIN,REWARD_LOSE

class SARSAAgent(QLearningAgent):
  
 
        
    #gamma=0.99, alpha=0.2
    def __init__(self,alpha=0.01, gamma=0.1, marker=None):
        """
            gamma: discount factor
            alpha: learning rate
            marker: The marker that the agent will use in the tic tac toe environment 
                   (if None, it should be set before starting learning)
        """
        super().__init__(alpha,gamma,marker)
        
        self.rnd = Seeds().SARSA_AGENT_SEED
        
        
        
        
        
                    
    def observe_reward(self,state,action,statePrime,reward):
        """
            Updates the Q-table (only if the agent is exploring
        """
        if self.exploring:
            allActionsPrime = self.environment.get_actions(statePrime)
        
           
            qValue, features = self.calcQTable(state,action,returnFeatures=True)
            self.exploring = False
            nextAction = self.select_action(statePrime)
            self.exploring = True
            if nextAction is None:
                V = 0.0
            else:
                V = self.calcQTable(statePrime,nextAction)
            expected = reward + self.gamma * V
            
            temporal_difference = expected - qValue         

            for i in range(len(self.qWeights)):
                self.qWeights[i] = self.qWeights[i] + self.alpha * (temporal_difference) * features[i]
            #self.qBias += self.alpha * temporal_difference * self.qBias
            #print(str(self.qWeights))#+ " - " + str(self.qBias))
            if self.epsilon > 0.05:
                self.epsilon /= 1.0001
            if self.alpha > 0.001:
                self.alpha /= 1.00001
            #print(self.alpha) 
            
    def best_action_deterministic(self,state):
        if state.is_game_over():
            return None
        return super().best_action_deterministic(state)
        
  
        
        
