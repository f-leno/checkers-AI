"""
    Random seeds for allowing repetition of the experiments (implemented as a Singleton)
    Author: Felipe Leno <f.leno@usp.br>
"""
import random
class Seeds:
    class __Seeds:
        Q_AGENT_SEED = 12345
        DQN_AGENT_SEED = 18345
        SARSA_AGENT_SEED = 12344
        SELFPLAY_AGENT_SEED = 22222
        RANDOM_AGENT_SEED = 11111
        EXPERT_AGENT_SEED = 54312
        SUBEXPERT_AGENT_SEED = 16875
        LENO_AGENT_SEED = 72810
        PROBMOD_AGENT_SEED = 96347
        NEURALNET_AGENT_SEED = 98130
        
               
        def __init__(self, trial):
            self.Q_AGENT_SEED += trial
            self.DQN_AGENT_SEED += trial
            self.SELFPLAY_AGENT_SEED += trial
            self.RANDOM_AGENT_SEED += trial
            self.EXPERT_AGENT_SEED += trial
            self.SUBEXPERT_AGENT_SEED += trial
            self.LENO_AGENT_SEED += trial
            self.PROBMOD_AGENT_SEED += trial
            self.NEURALNET_AGENT_SEED += trial
            self.SARSA_AGENT_SEED += trial
            
            
    instance = None
    def set_trial(self,trial):
        Seeds.instance = Seeds.__Seeds(trial)
        
    def __getattr__(self, name):
        seed = getattr(self.instance, name)
        obj = random.Random()
        obj.seed(seed)
        return obj
    
