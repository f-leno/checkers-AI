"""
A type of Class that will observe the transitions during the game and record any data of interest
    
    Author: Felipe Leno (f.leno@usp.br)
"""
import abc

class ExperimentRecorder(object):
    
    
    @abc.abstractmethod
    def track_step(self,state,actionX,actionO,statePrime,rewardX,rewardO):
        """
            This method will be called after each step in the environment
        """
        pass
    
    @abc.abstractmethod
    def end_episode(self,finalState):
        """
            Indicates the end of an episode
            --won: final state of the game
        """
        pass
    
    @abc.abstractmethod
    def end_learning(self):
        """
            Using for closing opened files, realeasing memory, etc.
        """
        pass
    
