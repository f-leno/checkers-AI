"""
    Implementation of a self play that plays against a version of itself that is most similar to
    the expert
    Author: Felipe Leno (f.leno@usp.br)

"""

from agents.qLearning import QLearningAgent
import copy
from randomSeeds.randomseeds import Seeds


class LenoSelfPlayAgent(QLearningAgent):
    
    fixedPolicy = False #If true, the agent never explores
    qTables = None #Library of previous Q-tables
    rnd = None
    
    #Maximum number of stored policies, if this number is exceeded, the less similar one is excluded
    maxCacheSize = 10
    
    #Storage from previously seen expert steps
    expertStorage = None
    
    
    def __init__(self,gamma=0.99, alpha=0.2, marker=None,qTableFollow = None, seed = None):
        """
            If the QTableFollow is not None, the agent follows the Q-table and never
            performs exploration
        """
        super().__init__(gamma, alpha, marker)
        self.rnd = Seeds().LENO_AGENT_SEED
        if seed is not None:
            self.rnd.seed(seed)
        self.qTables = []
        
        if qTableFollow is not None:
            self.qTable = qTableFollow
            self.fixedPolicy = True
        self.expertStorage = {}
    def score_policy(self,qTable,expertSteps):
        """
            Returns the percentage of states in which this Qtable would choose
            the same action as the expert
        """
        score = 0
        stTuples = expertSteps.keys()
        for stActTuple in stTuples:
            #Deterministic selection of actions
            state = stActTuple
            #state = self.process_state(state,marker)
            allActions = self.environment.get_actions(state)
            maxVal = -float('inf')
            bestAct = None
            for act in allActions:
                q = qTable.get((state,act),-float('inf'))
                if q > maxVal:
                    bestAct = [act]
                    maxVal = q
                elif maxVal == q:
                    if bestAct is None: #If the value does not exist in the q-table
                        bestAct = []
                    bestAct.append(act)
            chosen =  self.rnd.choice(bestAct) 
            if chosen in expertSteps[stActTuple]:
                score += 1
        return score
    
    def create_fictitious(self,expertSteps,marker):
        """
           Returns a fictitious agent with fixed policy
        """
        #Keeps a set of actions that have been already chosen by the expert
        for tupStAct in expertSteps:
            if tupStAct[0] not in self.expertStorage:
                self.expertStorage[(tupStAct[0])] = set()
            self.expertStorage[(tupStAct[0])].add(tupStAct[1])
            
            
        #Performs a copy of the current Qtable
        self.qTables.append(copy.deepcopy(self.qTable))
        #Calculates which of the previous policies are more similar to the expert policy
        policiesScore = [self.score_policy(qTab,self.expertStorage) for qTab in self.qTables]
        
                 
        #Selects the newest policy with highest score
        idx = len(policiesScore) - policiesScore[::-1].index(max(policiesScore)) - 1
        
        #Creates a new fictitious agent with random seed
        agent = LenoSelfPlayAgent(qTableFollow=self.qTables[idx], seed = self.rnd.randint(0,1000))
        
        #If the maximum size of stored Qtables is exceeded, the one with lowest score is eliminated
        if len(self.qTables) > self.maxCacheSize:
            del self.qTables[policiesScore.index(min(policiesScore))]
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
        return super().select_action(state)