"""
Agent base class for implementation of all algorithm.

This class defines the signature of the methods to interact with the environment
Methods to be implemented:
    --select_action(state): Returns the action to be executed by the agent.
    --observe_reward(state,action,statePrime,reward): Method to perform Q-updates or other needed updates
    
    Author: Felipe Leno (f.leno@usp.br)
"""
import abc
import game

class Agent(object):
    """ This is the base class for all agent implementations.
    """
    __metaclass__ = abc.ABCMeta
    
    environment = None 
    exploring = None #Is the Agent exploring?
    
    marker = None #Used for the tic tac toe environment
    
    
    def set_marker(self,marker):
        """
            Change the marker of the agent
        """
        self.marker = marker
    
    @abc.abstractmethod
    def select_action(self, state):
        """ When this method is called, the agent chooses an action. """
        pass
    
    def set_environment(self,environment,marker):
        """Connects to the domain environment"""
        self.environment = environment
        self.marker = marker 
        
    @abc.abstractmethod
    def observe_reward(self,state,action,statePrime,reward):
        """ After executing an action, the agent is informed about 
          the state-action-reward-state tuple """
        pass
    
    def set_exploring(self, exploring):
        """ The agent keeps track if it should explore in the current state (used for evaluations) """
        self.exploring = exploring
        
        
    def process_state(self,state, action):
        """
            Includes a variable in the state saying if the agent is X or O
        """
        return game.checkers_features(state, action)
        
        
        