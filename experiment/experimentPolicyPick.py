"""
Policy_Pick Experiment Class.
This class will play y rounds against the expert, while storing the actions taken by him.
Then, the <state,action> tuples will be given to the agent, that will return a fictitious players
to play against for x rounds
 
 Author: Felipe Leno (f.leno@usp.br) 

"""

from .experiment import ExperimentTicTacToe
 
class ExperimentPolicyPick(ExperimentTicTacToe):
    learningAgent = None
    expert = None
    
    x = None #Number of rounds against the expert
    y = None #number of rounds against the fictitious player
    
    currentRoundFict = None   #Number of rounds against the fictitious player
    currentRoundExp  = None   #number of rounds against the expertTrue
    nowFict          = None   #Is the agent facing the fictitious player now?
    
    learnersMarker = None #Marker of the learning agent
    
    expertSteps = None #Expert steps are recorded
    #totalEpis = 0 Used to debug if  the episode counting is correct
    #totalFict = 0
    #totalExp = 0
    
    def __init__(self,learning =None, expert = None, environment = None, totalEpisodes = None, 
                 totalSteps = None, experimentRecorder = None, showGUI = False, storeVideo=False,y=200,x=400,countFict = True):
        """
                learning: Learning agent that must have a create_fictitious() method 
                expert: Expert agent for this environment
                environment: The environment representing the task to be solved
                totalEpisodes: maximum number of episodes to execute (omit this parameter if stopping learning by number of steps)
                totalSteps: Maximum number of steps to execute (omit this parameter if stopping learning by episodes)
                showGUI: Should the game be displayed in a screen (much slower).
                storeVideo: If true, a video of the agent playing will be recorded 
                y = Number of rounds against expert
                x = Number of rounds against fictitious player
                countFic = Should the steps taken against fictitious player also count?
        """    
        super().__init__(agentO=learning,agentX=expert,environment=environment,totalEpisodes=totalEpisodes,
                                                   totalSteps=totalSteps,experimentRecorder=experimentRecorder,showGUI=showGUI,storeVideo=storeVideo)
        self.learningAgent = learning
        self.expert = expert
        self.x = x
        self.y = y
        self.currentRoundFict = 0
        self.currentRoundExp  = 0
        
        self.nowFict = False
        self.expertSteps = []
        self.learnersMarker = "O"
        self.countFict = countFict
        
        
    def run(self):
        """
            Runs the experiment according to the given parameters
        """
        #Give references to agents_checker and environment
        self.agentO.set_environment(self.environment, "O")
        self.agentX.set_environment(self.environment, "X")
        self.environment.set_agents(self.agentO,self.agentX)
        
        
        #Main perception-action loop
        while not self.stop_learning():
            #In this loop the turn of the two agents_checker will be processed, unless the game is over
            stateX = self.environment.get_state()
            #Get agent action
            actionX = self.agentX.select_action(stateX)
            #Applies state transition
            self.environment.step(actionX,"X")
               
            #If this is a terminal state, "O" lost the game and should be updated, 
            #If this is not the case, the agent makes its move
            if not self.environment.terminal_state():
                #Making the move...
                stateO = self.environment.get_state()
                self.recordStateO = stateO
                actionO = self.agentO.select_action(stateO)
                self.recordActionO = actionO
                self.environment.step(actionO,"O")
            else:
                stateO = self.recordStateO
                actionO = self.recordActionO
            #Updating...
            statePrime = self.environment.get_state()
            
            rewardX = self.environment.get_last_rewardX()
            #Update agent policy
            self.agentX.observe_reward(stateX,actionX,statePrime,rewardX)
            
            rewardO = self.environment.get_last_rewardO()
            self.agentO.observe_reward(stateO,actionO,statePrime,rewardO)
            
            #Record step, if required
            self.experimentRecorder.track_step(stateX,actionX,actionO,statePrime,rewardX,rewardO)
            
            #If the agent is playing against the expert, the steps are recorded
            if not self.nowFict:
                if self.learnersMarker == "X":
                    stateAgent = self.agentO.process_state(stateO)
                    expertTuple = (stateAgent,actionO)
                elif self.learnersMarker == "O":
                    stateAgent = self.agentX.process_state(stateX)
                    expertTuple = (stateAgent,actionX)
                self.expertSteps.append(expertTuple)
                
            if self.countFict or not self.nowFict:
                self.currentStep += 1
            #Check if the episode is over
            if self.environment.terminal_state():
                #self.totalEpis += 1 Used to debug if  the episode counting is correct
                #if self.nowFict:
                #    self.totalFict += 1
                #else:
                #    self.totalExp += 1
                #print("Total: "+str(self.totalEpis)+", Fict: "+str(self.totalFict)+", Exp: "+str(self.totalExp))
                if self.countFict or not self.nowFict:
                    self.currentEpisode += 1
                if self.nowFict:
                    self.currentRoundFict += 1
                else:
                    self.currentRoundExp += 1
                    
                self.experimentRecorder.end_episode(won = self.environment.won)
                self.environment.reset()                
                #Changes the learning agent side
                self.swap_agents()
                #Swap agent
                self.learnersMarker = "X" if self.learnersMarker== "O" else "O"
                
                self.check_change_fict()
                #Give references to agents_checker and environment
                self.agentO.set_environment(self.environment, "O")
                self.agentX.set_environment(self.environment, "X")
                self.environment.set_agents(self.agentO,self.agentX)
        
        #Reseting environment        
        self.currentEpisode = 0
        self.currentStep = 0
     
    def check_change_fict(self):
        """
            Check if it is time to change to the expert or to a new fictitious player
        """
        #The fictitious player was on and should be changed
        if self.currentRoundFict >= self.x:
            self.currentRoundFict = 0
            expertMarker = "X" if self.learnersMarker== "O" else "O"
            if expertMarker == "X":
                self.agentX = self.expert
            elif expertMarker == "O":
                self.agentO = self.expert
            self.nowFict = False
                
        #the expert was playing and it is time to change
        if self.currentRoundExp >= self.y and not self.nowFict:
            self.currentRoundExp = 0
            ficMarker = "X" if self.learnersMarker== "O" else "O"
            fictAgent = self.learningAgent.create_fictitious(self.expertSteps,self.learnersMarker)
            self.expertSteps = []
            if ficMarker == "X":
                self.agentX = fictAgent
            elif ficMarker == "O":
                self.agentO = fictAgent
            self.nowFict = True
            
        
            
#     def stop_learning(self):
#         """Checks if the experiment should stop now. Differently from the base class, here we check if games against the
#            fictitious player should count
#         """
#         stop = False
# 
#         if self.stopByEps:
#             numEpi = self.currentEpisode if self.countFict else self.episodeExpert
#             if numEpi >= self.totalEpisodes:
#                 stop = True
#         else:
#             numSte = self.currentStep
#             if numSte >= self.totalSteps:
#                 stop = True       
#             
#         return stop
    
        
        
