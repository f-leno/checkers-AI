"""
Tic Tac Toe Experiment Class.
This class will define which agents_checker will be learning in the environment,
whether if a GUI should be shown, and initiate all learning process.
 
 Author: Felipe Leno (f.leno@usp.br) 

"""

from agents.expertCheckersAgent import ExpertCheckersAgent

from environment.checkersEnvironment import CheckersEnvironment

from output.nullExperimentRecorder import NullExperimentRecorder


class ExperimentCheckers:
    
    agentO = None #Agent playing with O marker
    agentX = None #Agent playing with X marker
    totalEpisodes = None # Maximum number of Episodes (when applicable)
    totalSteps = None #Maximum number of Steps (when applicable)
    stopByEps = None 
    experimentRecorder = None #Class for saving experimental results
    showGUI = None #Should the GUI be shown?
    storeVideo = None #Should the video of the game play be recorded?
    
    currentEpisode = None #Current number of episodes
    currentStep = None #Current number of steps
    
    #In case the training process finishes during a game, and then agentX wins the game 
    #right after the return to the training phase, an error might happen without these variables 
    recordStateO = None
    recordActionO = None
    countFict = None    

    def __init__(self,agentO =None, agentX = None, environment = None, totalEpisodes = None, 
                 totalSteps = None, countFict = True, experimentRecorder = None, showGUI = False, storeVideo=False):
        """
            agentO,agentX: Agents playing with O and X markers. Both should be objects of the abstract class Agent. 
                Omit the parameter to create an expert agent.
            environment: The environment representing the task to be solved
            totalEpisodes: maximum number of episodes to execute (omit this parameter if stopping learning by number of steps)
            totalSteps: Maximum number of steps to execute (omit this parameter if stopping learning by episodes)
            countFict: Should the experiment recorder count steps taken when facing the simulated agent?
            showGUI: Should the game be displayed in a screen (much slower).
            storeVideo: If true, a video of the agent playing will be recorded 
        """
        #Basically, initializing variables
        if(totalEpisodes == None):
            self.stopByEps = False
        else:
            self.stopByEps = True

        if agentO == None:
            self.agentO = ExpertCheckersAgent()           
        else:
            self.agentO = agentO
        if agentX == None:
            self.agentX = ExpertCheckersAgent()
        else:
            self.agentX = agentX
            
        self.totalEpisodes = totalEpisodes
        self.totalSteps = totalSteps
        self.showGUI = showGUI
        self.storeVideo = storeVideo
        self.environment = CheckersEnvironment()
        if experimentRecorder == None:
            self.experimentRecorder = NullExperimentRecorder()
        else:
            self.experimentRecorder = experimentRecorder
        
        self.currentEpisode = 0
        self.currentStep = 0
        self.countFict = countFict
        
        
    def swap_agents(self):
        """
            The agents_checker swap side
        """
        aux = self.agentO
        self.agentO = self.agentX
        self.agentX = aux
        self.agentO.set_marker("O")
        self.agentX.set_marker("X")
    def run(self):
        """
            Runs the experiment according to the given parameters
        """
        #Give references to agents_checker and environment
        self.agentO.set_environment(self.environment, "O")
        self.agentX.set_environment(self.environment, "X")
        self.environment.set_agents(self.agentO,self.agentX)
        
        rewardO = 0
        actionO = None
        #Main perception-action loop
        while not self.stop_learning():
            doubleUpdate = False
            #In this loop the turn of the two agents_checker will be processed, unless the game is over
            stateX = self.environment.get_state()
            #Get agent action
            actionX = self.agentX.select_action(stateX)
            #Applies state transition
            self.environment.step(actionX)

            
            #If this is a terminal state, "O" lost the game and should be updated, 
            #If this is not the case, the agent makes its move
            stateO = self.environment.get_state() 
            if not self.environment.terminal_state():
                #Making the move...         
                actionO = self.agentO.select_action(stateO)
                self.environment.step(actionO)
                doubleUpdate = True

                

            #Updating...
            statePrime = self.environment.get_state()
            
            
            
            #Process rewards for agent O
            if self.recordStateO is not None:
                self.environment.process_rewards(pastState = self.recordStateO,currentState=stateO,agentMarker='O')
                rewardO = self.environment.get_last_rewardO()
                self.agentO.observe_reward(self.recordStateO,self.recordActionO,stateO,rewardO)
                
            self.recordStateO = stateO
            self.recordActionO = actionO    
            if self.environment.terminal_state() and doubleUpdate:
                self.environment.process_rewards(pastState = stateO,currentState=statePrime,agentMarker='O')
                rewardO = self.environment.get_last_rewardO()
                self.agentO.observe_reward(stateO,actionO,statePrime,rewardO)
                

                            
            #Process rewards for agent X
            self.environment.process_rewards(pastState = stateX,currentState=statePrime,agentMarker='X')
            rewardX = self.environment.get_last_rewardX()
            #Update agent policy
            self.agentX.observe_reward(stateX,actionX,statePrime,rewardX)
            
            
            #Record step, if required
            self.experimentRecorder.track_step(stateX,actionX,actionO,statePrime,rewardX,rewardO)
            
            self.currentStep += 1
            #Check if the episode is over
            if self.environment.terminal_state():
                self.currentEpisode += 1
                self.recordStateO = None
                self.recordActionO = None
                self.experimentRecorder.end_episode(finalState = self.environment.currentState)
                self.environment.reset()                
                #Changes the learning agent side
                self.swap_agents()
        
        #Reseting environment        
        self.currentEpisode = 0
        self.currentStep = 0
                
            
           
    def stop_learning(self):
        """Checks if the experiment should stop now"""
        stop = False
        if self.stopByEps:
            if self.currentEpisode >= self.totalEpisodes:
                stop = True
        else:
            if self.currentStep >= self.totalSteps:
                stop = True       
            
        return stop
        
        
        
        
        
            