"""
    Entry function for experiments in the Checkers domain.
    This domain has largely built upon https://github.com/SamRagusa/Checkers-Reinforcement-Learning
    
    Author: Felipe Leno (f.leno@usp.br)

"""
import argparse
import sys

from agents.expertCheckersAgent import ExpertCheckersAgent
from experiment.experiment import ExperimentCheckers
from output.discRewardExperimentRecorder import DiscRewardExperimentRecorder
from randomSeeds.randomseeds import Seeds
from agents.randomAgent import RandomAgent


def get_args():
    """Arguments for the experiment
            --learning_time: For how long should the agent be evaluated (in steps or episodes)
            --stopping_criterion: 'episode' for counting episodes or 'step' for counting steps
            --interval: Interval of evaluations (use the same criterion above)
            --trial: number of the repetition
            --duration: Number of evaluation games (always in episodes)
            --algorithm: Learning algorithm, given as "source:class", e.g., 'qLearning:QLearningAgent'
            --recording_mode: How should the experimental results be recorded?
                    - 'null' - Nothing is recorded
                    - 'reward' - Discounted reward is recorded
            --out: output folder
            --train_env: Which experiment will be executed? The choice of learning algorithm must correspond
                         to the parameter given here.
                    - 'regular' - The agent plays against the expert normally
                    - 'policy_pick' - some rounds are played against the expert while storing the chosen actions. Then, the agent
                                      chooses one of its previous policies that are similar to the expert, and play against it for some
                                      rounds.
            
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--learning_time',type=int, default=50000)
    parser.add_argument('-s','--stopping_criterion',choices=['episode,step'],default="step")
    parser.add_argument('-i','--interval',type=int,  default=500)
    parser.add_argument('-t','--trial',type=int,  default=1) 
    parser.add_argument('-d','--duration',type=int, default=5)
    parser.add_argument('-a','--algorithm',default='qLearning:QLearningAgent')
    parser.add_argument('-r','--recording_mode',choices=['null','reward'],default="null")
    parser.add_argument('-o','--out',default='./results/')
    parser.add_argument('-e','--train_env',choices=['regular','policy_pick','transfer_fw','transfer_fs','self_play'],default="regular")
    parser.add_argument('-c','--count_fict',choices=['yes','no','true','false','t','f','y','n','1','0'], default='1')
    parser.add_argument('-y','--number_expert',type=int, default=200)
    parser.add_argument('-x','--number_simulated',type=int, default=400)
    parser.add_argument('-ex','--expert_optimality',type=float, default=1.0)  
    
    parameter = parser.parse_args()
    
    parameter.count_fict = parameter.count_fict in ('yes', 'true', 't', 'y', '1')
    
    
    return parameter

def build_agent(parameter):
    """Builds and returns the agent object as specified by the arguments"""
    
    
    #split in source, agent class
    fileName,className = parameter.algorithm.split(":")
    #Try to import agent class
    try:
        AgentClass = getattr(
            __import__('agents.' + fileName,
                      fromlist=[className]),
                      className)
    except ImportError as error:
        print(error)
        sys.stderr.write("ERROR importing python module: " +parameter.algorithm + "\n")
        sys.exit(1)
        
    #If everything is OK, creates the object
    agent = AgentClass()
    
    return agent

def build_env(learningAgent,expert):
    """ Builds and returns the Experiment Environment for learning"""
    
    parameter = get_args()
    if parameter.train_env == "regular":
        if parameter.stopping_criterion == 'episode':
            trainingEnv = ExperimentCheckers(agentO = learningAgent, agentX = expert,totalEpisodes = parameter.interval,countFict = parameter.count_fict)
        elif parameter.stopping_criterion == 'step':
            trainingEnv = ExperimentCheckers(agentO = learningAgent, agentX = expert,totalSteps = parameter.interval,countFict = parameter.count_fict)
    elif parameter.train_env in ["policy_pick","transfer_fs","transfer_fw"]:
        if parameter.stopping_criterion == 'episode':
            trainingEnv = ExperimentCheckers(learning = learningAgent, expert = expert,totalEpisodes = parameter.interval,countFict = parameter.count_fict, y=parameter.number_expert,x=parameter.number_simulated)
        elif parameter.stopping_criterion == 'step':
            trainingEnv = ExperimentCheckers(learning = learningAgent, expert = expert,totalSteps = parameter.interval,countFict = parameter.count_fict, y=parameter.number_expert,x=parameter.number_simulated)
        
    return trainingEnv

def main():
    parameter = get_args()
    
    #Change Seeds according to trial
    Seeds().set_trial(parameter.trial)
    
    print("Starting Experiment:")
    print(parameter)
    #Creates a folder with the name of the algorithm
    parameter.out += parameter.algorithm.split(":")[0] 
    if parameter.count_fict:
        parameter.out += "_countF" 
    else:
        parameter.out += "_noF"
    if parameter.expert_optimality < 1.0:
        parameter.out += "_" + str(parameter.expert_optimality)
    
    if parameter.train_env in ["transfer_fs","transfer_fw"]:
        parameter.out += "_"+parameter.train_env 
        
    parameter.out +=  "/"
    

    
    #Creates references for the agents_checker
    learningAgent = build_agent(parameter)
    
    if parameter.expert_optimality < 1.0 and parameter.train_env != "transfer_fs":
        expertTrain = SuboptimalExpert(parameter.expert_optimality)
    else:
        expertTrain = RandomAgent()#ExpertCheckersAgent() 
        
    
        
    if parameter.train_env == "transfer_fw":
        expertEval = RandomAgent()#ExpertCheckersAgent()
    elif parameter.train_env == "transfer_fs":
        expertEval = SuboptimalExpert(parameter.expert_optimality)
    else:
        expertEval = expertTrain
        
    
   
    
    
        
    if parameter.recording_mode == 'reward':
        recorder = DiscRewardExperimentRecorder(gamma=learningAgent.gamma, agentMarker = "O", 
                                               interval = parameter.interval,resultFolder = parameter.out, trial = parameter.trial)
    else:
        recorder = None
    #Creates the environments to be used for training and evaluation
    #Those environment will run the learning process according to the time defined by the parameters
    #of this source code
    evalEnv = ExperimentCheckers(agentO = learningAgent, agentX = expertEval, totalEpisodes = parameter.duration, 
                 experimentRecorder=recorder)#totalSteps = None, experimentRecorder = None, showGUI = False, storeVideo=False)
    
    
    trainingEnv = build_env(learningAgent,expertTrain)
    
    
    #Calculates how many times the learning experiment will be executed
    training_sessions = int(parameter.learning_time / parameter.interval)
    
    
    print("EVALUATION: " + str(0) + parameter.stopping_criterion)
    #Initial evaluation with 0 training
    learningAgent.exploring = False
    #expert.exploring = False
    evalEnv.run()
    recorder.finish_experiment()
    
    
    for i in range(training_sessions):
        #First trains the agent, then evaluates
        print("TRAINING: " + str(i*parameter.interval) + parameter.stopping_criterion)
        learningAgent.exploring = True
        #expert.exploring = True
        trainingEnv.run()
        print("EVALUATION: " + str((i+1)*parameter.interval) + parameter.stopping_criterion)
        learningAgent.exploring = False
        #expert.exploring = False
        evalEnv.run()
        recorder.finish_experiment()
        
    print("Finished")
    recorder.end_learning()
    


        
        
        
    
    
    
    
if __name__ == '__main__':
    main()
    
 
