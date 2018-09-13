"""
    Entry function for experiments in the Checkers domain.
    This domain has largely built upon https://github.com/SamRagusa/Checkers-Reinforcement-Learning
    
    Author: Felipe Leno (f.leno@usp.br)

"""

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
    parser.add_argument('-l','--learning_time',type=int, default=15000)
    parser.add_argument('-s','--stopping_criterion',choices=['episode,step'],default="step")
    parser.add_argument('-i','--interval',type=int,  default=150)
    parser.add_argument('-t','--trial',type=int,  default=1) 
    parser.add_argument('-d','--duration',type=int, default=16)
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
