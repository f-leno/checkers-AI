"""
    Records the discounted reward for the agent during the experiments.
    
    Author: Felipe Leno <f.leno@usp.br>
"""
from output.experimentRecorder import ExperimentRecorder
import csv
import os

class DiscRewardExperimentRecorder(ExperimentRecorder):
    agentMarker = None #Keeps the marker of the agent to track
    gamma = None #Discount factor
    currentGamna = None #Used to calculate the discounted reward
    currentReward = None #Sum of rewards for the current episode
    sumRewards = None #All sum of rewards
    interval = None #Number of learning episodes/steps between each finish_experiment() call
    currentEvaluation = None #Keeping track of how many training episodes/steps were executed
    resultFilePath = None
    csvFile = None
    csvWriter = None
    
    #Statistics for the games
    numberWins = None
    numberLoses = None
    numberTies = None
    
    gameStore = True #should the games be recorded (for debbuging purposes)
    gamesFile = None
    gamesFilePath = None
    
    useDiscounted = None
    
    def __init__(self,gamma,agentMarker,interval,resultFolder,trial,useDiscounted = False):
        """
            gamma: Discount factor
            agentMarker: Initial marker of the agent, which is supposed to change every episode
            interval: interval between evaluations (each time finish_experiment() is called)
            resultFolder: Folder for the result
            trial: number of this repetition
        """
        self.agentMarker = agentMarker
        self.gamma = gamma
        self.currentGamma = 1.0
        self.sumRewards = []
        self.interval = interval
        self.currentEvaluation = 0
        self.currentReward = 0.0 
        self.numberWins = 0
        self.numberLoses = 0
        self.numberTies = 0
        
        if not os.path.exists(resultFolder):
            os.makedirs(resultFolder)
        self.resultFilePath = resultFolder + "discRew" + str(trial)+ ".csv"
        if self.gameStore:
            self.gamesFilePath = resultFolder + "games.txt"
            self.gamesFile = open(self.gamesFilePath, "w")
        self.csvFile = open(self.resultFilePath, "w")
        self.csvWriter = csv.writer(self.csvFile)

        self.csvWriter.writerow(("time","reward","W","T","L"))
        self.csvFile.flush()
        
        self.useDiscounted = useDiscounted

        
    def end_learning(self):
        """
            When everything finishes, the file is closed
        """
        self.csvFile.close()
        if self.gameStore:
            self.gamesFile.close()
    
    def track_step(self,state,actionX,actionO,statePrime,rewardX,rewardO):
        """
            Records the discounted reward for the experiment
        """
        
        rewardAgent = rewardX if self.agentMarker=="X" else rewardO
        self.currentReward += float(rewardAgent)*self.currentGamma
        
        #Updating discount fcurrentEvaluationactor
        if self.useDiscounted:
            self.currentGamma *= self.gamma
        if self.gameStore:
            sentence = "State_inic - "+self.agentMarker+"\n" + state.print_board(True)+ "\n actionX: " + str(actionX)+ ", actionO: " + str(actionO)
            sentence += "\n State final\n: "+ statePrime.print_board(True)+ "\nReward:" + str(rewardX) +","+str(rewardO)+"\n"
            self.gamesFile.write(sentence)
            self.gamesFile.flush()
        
        #print ( "Step - " + str(state))
        #print ("actions: " + str(actionX) + "," + str(actionO))
        #print ("rewards: "+ str(rewardX) + "," + str(rewardO))
        #print(statePrime)

        
        
        
        
    def end_episode(self, finalState):
        """
            Agent marker changes and reward for the experiment is stored
            --won: "X" or "O" if one of the agent won, None if tie
        """
        if not finalState.is_game_over():
            self.numberTies += 1
        elif finalState.is_first_agent_win() and self.agentMarker == 'X' or \
                finalState.is_second_agent_win() and self.agentMarker == 'O':
            self.numberWins += 1
        else:
            self.numberLoses +=1
       
        self.agentMarker = "O" if self.agentMarker=="X" else "X"
        self.sumRewards.append(self.currentReward)
        self.currentGamma = 1.0
        self.currentReward = 0.0
        
        
        
    def finish_experiment(self):
        """
           The experiment is over, record what is needed
        """
        #record result
        value = sum(self.sumRewards)        
        self.csvWriter.writerow((self.currentEvaluation,"{:.4f}".format(value),
                                 str(self.numberWins),str(self.numberTies),str(self.numberLoses) ))
        self.csvFile.flush()
        if self.gameStore:
            self.gamesFile.write("-----Finish Match-------\n\n")
            self.gamesFile.flush()
        self.sumRewards = []
        self.numberWins = 0
        self.numberLoses = 0
        self.numberTies = 0
        self.currentEvaluation += self.interval

        
        
        
        
        