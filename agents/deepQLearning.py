"""
    Implementation of Q-Learning with Linear Q-function approximation
    
    Author: Felipe Leno (f.leno@usp.br)

"""

import math

from agents.qLearning import QLearningAgent

import numpy as np
from randomSeeds.randomseeds import Seeds
from game import CHECKERS_FEATURE_COUNT
from environment.checkersEnvironment import REWARD_WIN,REWARD_LOSE
import keras.layers
from keras.models import Model
from keras.layers import Input
from keras import layers


class DeepQLearningAgent(QLearningAgent):
    
    batch = None
    miniBatchSize = None
    
    updateFrequency = None
    
    batchSize = None
    currentStep = None
    
    
    #gamma=0.99, alpha=0.2
    def __init__(self,alpha=0.01, gamma=0.1, marker=None, batchSize=10000, miniBatchSize = 100,updateFrequency=100):
        """
            gamma: discount factor
            alpha: learning rate
            marker: The marker that the agent will use in the tic tac toe environment 
                   (if None, it should be set before starting learning)
        """
        super().__init__(alpha=alpha, gamma=gamma, marker=marker)    
            
        self.rnd = Seeds().DQN_AGENT_SEED
        from numpy.random import seed
        seed(self.rnd.randint(1,10000))
        from tensorflow import set_random_seed
        set_random_seed(self.rnd.randint(1,10000))
        
        self.batch = []
        self.batchSize = batchSize
        self.miniBatchSize = miniBatchSize
        self.updateFrequency = updateFrequency
        self.currentStep = 0
        
        self.init_network()
        
        
        
    def init_network(self):
        """
            Builds the neural network for the agent
        """
        hiddenNeurons = 20
        
        inp = Input(shape=(CHECKERS_FEATURE_COUNT,))
        
        net = layers.Dense(hiddenNeurons, activation="relu")(inp)
        net = layers.Dropout(0.2, seed = self.rnd.randint(1,10000)) (net)
        net = layers.Dense(hiddenNeurons, activation="sigmoid")(net)
        net = layers.Dense(1, activation="sigmoid")(net)
        
        self.network = Model(inputs=inp, outputs=net)
        self.target = keras.models.clone_model(self.network)
        
        self.cost = keras.losses.mean_squared_error
        self.network.compile(optimizer=keras.optimizers.Adam(),loss=self.cost)
        
        
        #self.network.summary()
        
    def update_network(self,miniBatch):
        featuresOnBatch = np.array([x[0] for x in miniBatch])
        
        #Targets on target network
        targets = []
        for (features,statePrime,reward) in miniBatch:
            if statePrime.is_terminal():
                targets.append(reward)
            else:
                actions = self.environment.get_actions(statePrime)
                maxVal = self.get_max_Q_value(statePrime,allActions=actions,network=self.target)
                targets.append(reward + self.gamma*maxVal)
        targets = np.array(targets)
                
        #Q values on network being updated
        q_values = self.network.predict_on_batch(featuresOnBatch)
        
        
        #Optimization process
        deltas = self.cost.get_errors(targets,q_values)
        self.network.bprop(deltas)
        self.optimizer.optimize(self.network.layers_to_optimize) 
        
    def update_target_network(self):
        self.target.set_weights(self.network.get_weights())
        
             
                    
    def observe_reward(self,state,action,statePrime,reward):
        """
            Updates the Q-table (only if the agent is exploring
        """
        if self.exploring:
            #Updating batch
            features = self.process_state(state,action)
            self.batch.append((features,statePrime,reward))
            if len(self.batch > self.batchSize):
                del self.batch[0]
                
            miniBatch = self.rnd.sample(self.batch, self.miniBatchSize)
            self.update_network(miniBatch)
                
            if self.epsilon > 0.05:
                self.epsilon /= 1.0001
            if self.alpha > 0.001:
                self.alpha /= 1.00001
                
            if self.currentStep % self.updateFrequency == 0:
                self.update_target_network()
            self.currentStep += 1
            #print(self.alpha) 
            
    def calcQTable(self,state,action,returnFeatures=False,network=None):             
        """Returns one value from the Qtable"""
        if network is None:
            network = self.target
        features = self.process_state(state,action)
        qValue = network.predict(features) #+ self.qBias
        
        if returnFeatures:
            return qValue,features
        
        return qValue  
    
    def get_max_Q_value(self,state,allActions,network=None):
        """
            returns max(Q) for all actions given a state
        """
        if network is None:
            network = self.target
        
        values = [self.calcQTable(state, action,network=network) for action in allActions]
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
        
        
