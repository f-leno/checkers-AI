"""
    Simple implementation of QLearning for the tic tac toe domain
    
    Author: Felipe Leno (f.leno@usp.br)

"""

from agents.agent import Agent

from randomSeeds.randomseeds import Seeds
import math
from agents.expertTicTacToeAgent import ExpertTicTacToeAgent

class QLearningAgent(Agent):
    gamma = None
    alpha = None
    
    qTable = None
    initQ = None #Value to initiate Q-table
    
    USE_EPSILON_GREEDY = True #If false, uses Boltzmann exploration
    epsilon = 0.1
    

    
    T = None #Current temperature for Boltzmann exploration
    tempInit = 1.0/2 #Initial temperature for Boltzmann exploration
    tempFinal = 1.0/50 #Final temperature for Boltzmann exploration
    
    rnd = None
    
   
 
        
    
    def __init__(self,gamma=0.99, alpha=0.2, marker=None):
        """
            gamma: discount factor
            alpha: learning rate
            marker: The marker that the agent will use in the tic tac toe environment 
                   (if None, it should be set before starting learning)
        """
        self.gamma = gamma
        self.alpha = alpha
        self.marker = marker
        
        self.qTable = {}
        self.initQ = 0.0
        self.T = self.tempInit;
        self.rnd = Seeds().Q_AGENT_SEED
        
                
        
    
    def observe_reward(self,state,action,statePrime,reward):
        """
            Updates the Q-table (only if the agent is exploring
        """
        if self.exploring:
            #print("UPDATING")
            state = self.process_state(state)
            allActionsPrime = self.environment.get_actions(statePrime)
            statePrime = self.process_state(statePrime)
            #Regular q-update
            qValue= self.readQTable(state,action)
            V = self.get_max_Q_value(statePrime,allActionsPrime)        
            newQ = qValue + self.alpha * (reward + self.gamma * V - qValue)
            self.qTable[(state,action)] = newQ
            
    def readQTable(self,state,action):             
        """Returns one value from the Qtable"""
        if not (state,action) in self.qTable:
            self.qTable[(state,action)] = self.initQ
        return self.qTable[(state,action)]  
    
    def best_action_deterministic(self,state):
        allActions = self.environment.get_actions(state)
        maxVal = -float('inf')
        bestAct = None
        for act in allActions:
            q = self.readQTable(state, act)
            if q > maxVal:
                bestAct = [act]
                maxVal = q
            elif maxVal == q:
                bestAct.append(act)
        return self.rnd.choice(bestAct)
    def select_action(self, state):
        """ When this method is called, the agent executes an action based on its Q-table 
            Boltzmann Exploration is used    def create_fictitious(self,expertSteps):
        """
        if self.USE_EPSILON_GREEDY:
            return self.select_action_epsilon_greedy(state)
        else:
            return self.select_action_boltzmann(state)

    def select_action_boltzmann(self,state):
        #Check here
        state = self.process_state(state)
        allActions = self.environment.get_actions()
        #Boltzmann exploration strategy
        valueActions = []
        sumActions = 0
            
        for action in allActions:
            qValue = self.readQTable(state,action)
            vBoltz = math.pow(math.e,qValue/self.T)
            valueActions.append(vBoltz)
            sumActions += vBoltz
            
        probAct = [x / sumActions for x in valueActions]
          
        rndVal = self.rnd.random()
            
        sumProbs = 0
        i=-1
            
        while sumProbs <= rndVal:
            i = i+1
            sumProbs += probAct[i]
            
        #Apply decay
        if self.T > self.tempFinal and self.exploring:
            self.T -= (self.tempInit - self.tempFinal)/(100000);
            
        return allActions[i]
    
    def select_action_epsilon_greedy(self,state):
        """
            Applies the epsilon greedy exploration when the agent is exploring,
            and chooses deterministically 
        """
        state = self.process_state(state)
        
        randv = self.rnd.random()
        if self.exploring and randv < self.epsilon:
            return self.rnd.choice(self.environment.get_actions())
        
        #If not random explorating, the action with best value is returned
        return self.best_action_deterministic(state) 
        
        
    def get_max_Q_value(self,state,allActions):
        """
            returns max(Q) for all actions given a state
        """
        
        values = [self.readQTable(state, action) for action in allActions]
        #Terminal states don't have applicable actions
        if len(values) == 0:
            return 0
        #Maximum Q value
        v = max(values)
        return v

    def create_fictitious(self,expertSteps,marker=None):
        """
            Regular Q-learning always plays against the expert
        """
        #This function is called before the agent is set to the new Marker.
        if self.marker == "X":
            return self.environment.agentX
        
        return self.environment.agentO
        
        