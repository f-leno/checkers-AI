"""
    This class does not record any information
"""
from output.experimentRecorder import ExperimentRecorder
class NullExperimentRecorder(ExperimentRecorder):
    def track_step(self,state,actionX,actionO,statePrime,rewardX,rewardO):
        pass
    def end_episode(self,finalState):
        pass
    def end_learning(self):
        pass