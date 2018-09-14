"""
    Class that models the opponent by simply counting the observed actions for each state, and applying random actions when the state was
    not visited before.

"""

from agents.modelQLearning import ModelQLearningAgent
from randomSeeds.randomseeds import Seeds
import numpy as np
import copy as cp

class ProbModelQLearning(ModelQLearningAgent):
    
    followModel = False
    
    def __init__(self,gamma=0.99, alpha=0.2, marker=None, model = None, seed = None):
        """
            If model is not None, the agent follows the model and never
            performs exploration
        """
        super().__init__(gamma, alpha, marker)

        self.rnd = Seeds().PROBMOD_AGENT_SEED
        if seed is not None:
            self.rnd.seed(seed)
        np.random.seed(self.rnd.randint(1,10000))
            
        if model is not None:
            self.model = model
            self.followModel = True
            self.exploring = False
            
    def init_model(self):
        """
            The model is a simple dictionary that will count the chosen actions
        """
        return {}        
        
    def update_model(self,expertSteps,marker):
        """
            The model keeps counters of chosen actions, which are updated here
        """
        #Updates the model
        for tupStAct in expertSteps:
            state = tupStAct[0]
            if (state,tupStAct[1]) not in self.model:
                self.model[(state,tupStAct[1])] = 0
            self.model[(state,tupStAct[1])] += 1
            
    def agent_from_model(self,marker):
        """
            Creates an agent from the model
        """
        model = cp.deepcopy(self.model)
        return ProbModelQLearning(model=model, seed = self.rnd.randint(1,10000))
    
    def action_from_model(self,state):
        """
            selects an action according to the observed decisions of the expert agent
        """
        state = self.process_state(state)
        actions = self.environment.get_actions(state)
        visits = []
        
        #Searches for the number of times that each action was chosen the current state
        for act in actions:
            v = self.model.get((state,act),0)
            visits.append(v)
        
        totalVals = sum(visits)
        if totalVals == 0:
            return self.rnd.choice(actions)
        
        actProbs = [x / totalVals for x in visits]
        chosenIdx = np.random.choice(len(actions),p=actProbs)
        
        return actions[chosenIdx]
            
        
        
        
            
    def select_action(self, state):
        if self.followModel:
            return self.action_from_model(state)
        return super().select_action(state)
    
           
           
    