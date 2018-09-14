"""
    Abstract class for agents that build a model of the opponent.
    Concrete classes must implement: (i) how to initialize the model; (ii) how to update the model with new instances;
    (iii) how to create a fictitious agent from the model.
    Author: Felipe Leno (f.leno@usp.br)

"""

from agents.qLearning import QLearningAgent
import abc


class ModelQLearningAgent(QLearningAgent):
    __metaclass__ = abc.ABCMeta
     
    
    model = None #Model for the opponent agent 
    
    @abc.abstractmethod
    def init_model(self):
        """
            The agent initializes the model here, if needed
        """
        pass
    @abc.abstractmethod
    def update_model(self,expertSteps,marker):
        """
            Given a set of instances, the agent should update its model of the expert
        """
        pass
    @abc.abstractmethod
    def agent_from_model(self,marker):
        """
            Creates a fictitious opponent from the model
        """
        pass
    
    
    def __init__(self,gamma=0.99, alpha=0.2, marker=None):
        super().__init__(gamma, alpha, marker)
        self.model = self.init_model()
        
      
    def create_fictitious(self,expertSteps,marker=None):
        """
           Returns a fictitious agent with fixed policy
        """
        #Updates the model with the new information
        self.update_model(expertSteps,marker)
        
        #Creates and returns an agent from the model
        return self.agent_from_model(marker)

   
   