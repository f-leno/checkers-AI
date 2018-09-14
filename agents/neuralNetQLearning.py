"""
    Class that models the opponent by simply counting the observed actions for each state, and applying random actions when the state was
    not visited before.

"""

from agents.modelQLearning import ModelQLearningAgent
from randomSeeds.randomseeds import Seeds
import numpy as np


import tensorflow as tf


class NeuralNetModQLearning(ModelQLearningAgent):
    
    followModel = False
    
    
    modelAlpha = 0.2 #Learning rate of the model   
    initOp = None #initializer for tf session
    y_hat = None #used for predictions
    X = None
    y= None
    predict = None
    initialModel = None
    W1 = None
    W2 = None
    b1 = None
    b2 = None
    session=None
    
    examplesChache = []
    maxExamples = 500
    
    
    
    
    
    
    def __init__(self,gamma=0.99, alpha=0.2, marker=None, model = None, seed = None):
        """
            If model is not None, the agent follows the model and never
            performs exploration
        """
        #This should be executed before the constructor of the super class
        if model is not None:
            self.initialModel = model
        super().__init__(gamma, alpha, marker)

        self.rnd = Seeds().NEURALNET_AGENT_SEED
        if seed is not None:
            self.rnd.seed(seed)
        np.random.seed(self.rnd.randint(1,10000))
        if model is not None:   
            self.followModel = True
            self.exploring = False 
            
    def init_model(self):
        """
            The model here is a tensorflow neural network. Everything will be prepared for later use.
        """

        
        
        num_features = 9
        num_actions = 9
        
        num_hidden_neurons = 20
        with tf.Graph().as_default() as g:
            #X will be the state variables (What is inside each of the 9 positions
            self.X = tf.placeholder(tf.float32, [None,9], name = 'X')
            #y is the action, 9 positions because of the one-hot encoding
            self.y = tf.placeholder(tf.float32, [None,9], name = "y")
            
            
        
    #        if self.initModel is not None:
    #            initW1 = self.initModel.W1
    #            initb1 = self.initModel.b1
    #            initW2 = self.initModel.W2
    #            initb2 = self.initModel.b2
                #weights and biases
    #            self.W1 = tf.Variable(initW1)
    #            self.b1 = tf.Variable(initb1)
    #            self.W2 = tf.Variable(initW2)
    #            self.b2 = tf.Variable(initb2)
    #        else:
            self.W1 = tf.Variable(tf.random_uniform([num_features,num_hidden_neurons],seed = self.rnd.randint(0,1000), 
                                                   minval = 0.0001, maxval=0.1), name='W1')
            self.b1 = tf.Variable(tf.random_uniform([num_hidden_neurons], seed = self.rnd.randint(0,1000)), name='b1')
            self.W2 = tf.Variable(tf.random_uniform([num_hidden_neurons,num_actions],seed = self.rnd.randint(0,1000), 
                                                   minval = 0.0001, maxval=0.1), name='W2')
            self.b2 = tf.Variable(tf.random_uniform([num_actions], seed = self.rnd.randint(0,1000)), name='b2')
            
            #Calculating the output of hidden layers
            hidden_out = tf.add(tf.matmul(self.X,self.W1),self.b1)
            hidden_out = tf.nn.sigmoid(hidden_out)
            
            self.y_hat = tf.nn.softmax(tf.add(tf.matmul(hidden_out, self.W2), self.b2))
            
            #self.predict = tf.argmax(self.y_hat,axis=1)
            self.predict = tf.multinomial(self.y_hat, seed = self.rnd.randint(0,1000), num_samples=1)
            y_clipped = tf.clip_by_value(self.y_hat, 1e-10, 0.9999999)
            #Cost function (cross-entropy)
            self.cost = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(y_clipped)
                             + (1 - self.y) * tf.log(1 - y_clipped), axis=1))
                   
            # add an optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.modelAlpha).minimize(self.cost)
            
            self.initOp = tf.global_variables_initializer()
            
            self.saver = tf.train.Saver()
        
        self.session = tf.Session(graph=g)
            
        self.session.run(self.initOp)
        
        if self.initialModel is not None:
            self.saver.restore(self.session,self.initialModel)
        
    def update_model(self,expertSteps,marker):
        """
            The model keeps counters of chosen actions, which are updated here
        """
        if len(self.examplesChache) + len(expertSteps) > self.maxExamples:
            #Open space in the batch for new samples
            del self.examplesChache[0:len(expertSteps) - (self.maxExamples - len(self.examplesChache))]
        self.examplesChache.extend(expertSteps)
        expertSteps = self.examplesChache 

        X = self.states_to_float(np.array(expertSteps)[:,0])
        
        #print(X)
        y = self.convert_actions(np.array(expertSteps)[:,1])       

        #Trains with the data for 10 epochs
        epochs = 10
        # start the session
        sess = self.session
        
        #print(len(expertSteps))
        for epoch in range(epochs):
            avg_cost = 0
            
            for i in range(len(expertSteps)):
                _, c = sess.run([self.optimizer,self.cost],feed_dict={self.X: X, self.y: y})
                avg_cost += c / len(expertSteps)
            #print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        #print(sess.run(self.W1))
                
        
    def states_to_float(self,states):
        """
            Convert the state in string to float
            0 == ., 1==N, 2==S
        """
        procStates = np.zeros((len(states),len(states[0])),dtype=np.float32)
        i=0
        for state in states:
            procStates[i,:] = [0. if x=='.' else 1. if x=='N' else 2. for x in state ]
            i += 1
        
        
        return procStates

    def convert_actions(self,actions):
        """
            Converts the action into the 0-9 interval and back if needed
        """
        convertedActions = []
        for act in actions:
            convertedActions.append(act[1]*3 + act[0])
        convertedActions = np.asarray(convertedActions)
        #Convert to one-hot
        num_actions = 9
        convertedActions = convertedActions.reshape(-1)
        convertedActions = np.eye(num_actions, dtype=np.float32)[convertedActions]
        return convertedActions
                
    def agent_from_model(self,marker):
        """
            Creates an agent from the model
        """
        model = "/tmp/model.save"
        #print("Will save")
        self.saver.save(self.session,model)
        
        #print("Saved")
        return NeuralNetModQLearning(model=model, seed = self.rnd.randint(1,10000))
    
    def action_from_model(self,state):
        """
            selects an action according to the observed decisions of the expert agent
        """
        state = self.process_state(state)
        X = np.array(self.states_to_float([np.array(state)]),dtype=np.float32)
        #print(X.dtype)
        #print(X)
        #print(X.shape)
        import math
        
        acts = self.environment.get_actions(state)
        
        sess = self.session
            
        act = float('inf')
        
        while not act in acts:
            act = sess.run([self.predict],feed_dict={self.X: X})
        
            #print(act[0][0])
            act = (math.floor(act[0][0]%3), math.floor(act[0][0]/3))
            
        #print(act)
        return act
        
        
        
            
        
        
        
            
    def select_action(self, state):
        if self.followModel:
            
            return self.action_from_model(state)
        return super().select_action(state)
    
           
           
    