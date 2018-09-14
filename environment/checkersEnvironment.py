"""
    Checkers environment. Mostly an adapter to the original implementation
    https://github.com/VarunRaval48/checkers-AI
    
    Author: Felipe Leno <f.leno@usp.br>
"""

#TODO: Check if the reward calculation is correct

import checkers

REWARD_WIN = 500
REWARD_LOSE = -500
REWARD_DEFAULT = -0.1


class CheckersEnvironment():
    
    #Agents controlling the X and O marks
    agentX = None
    agentY = None
    
    #Variable to store current state and return it more quickly
    currentState = None
    lastState = None

    
    rewardX = None
    rewardO = None 

    terminal = None
    num_moves = None #Keep track of the maximum number of moves in the board
    
    checkersControl = None #Reference to the original environment codification
    
    def terminal_state(self):
        return self.terminal
    
    def get_env_name(self):
        return "Checkers"
    
    def __init__(self):
        """
            Constructor for the environment. Mostly responsible for initializing the board
        """
        self.reset()
        
    def reset(self):
        self.checkersControl = checkers.ClassicGameRules().new_game(first_agent=self.agentX, second_agent=self.agentO, first_agent_turn=True, quiet=True)
        #The current state is updated
        self.process_state()
        self.lastState = self.currentState
        
        self.rewardO = REWARD_DEFAULT
        self.rewardX = REWARD_DEFAULT
        
        self.terminal = False
        self.num_moves = 0
    
    def set_agents(self,agentO,agentX):
        """
            Stores a reference to the agents
        """
        self.agentX = agentX
        self.agentO = agentO
    
    def get_state(self):
        """
            returns the current state
        """
        return self.currentState
    def step(self,action):
        """
            Applies the next action in the environment:
            action : position selected by agent
        """
        next_game_state = self.checkersControl.game_state.generate_successor(action)
        self.checkersControl.game_state = next_game_state
        #State transition
        self.num_moves += 1
        
        self.process_state()
        #Process rewards
        self.process_rewards()
        
    def process_state(self):
        """ 
             Stores the current state in the currentState variable, and checks if it is a terminal state
        """
        self.lastState = self.currentState
        self.currentState = self.checkersControl.game_state
        self.terminal = self.currentState.is_game_over() or self.num_moves > self.checkersControl.rules.max_moves
            
    def process_rewards(self):  
        """
            Prepares rewards for both agents
        """ 
        if self.terminal and self.num_moves <= self.checkersControl.rules.max_moves:
            self.rewardX = REWARD_WIN if self.currentState.is_first_agent_win() else REWARD_LOSE
            self.rewardO = REWARD_WIN if self.currentState.is_second_agent_win() else REWARD_LOSE
        else:

            #Calculating Reward for agent 1
            agent_ind = 0 
            oppn_ind = 1 

            num_pieces_list = self.lastState.get_pieces_and_kings()
            agent_pawns = num_pieces_list[agent_ind]
            agent_kings = num_pieces_list[agent_ind + 2]
            oppn_pawns = num_pieces_list[oppn_ind]
            oppn_kings = num_pieces_list[oppn_ind + 2]

            num_pieces_list_n = self.currentState.get_pieces_and_kings()
            agent_pawns_n = num_pieces_list_n[agent_ind]
            agent_kings_n = num_pieces_list_n[agent_ind + 2]
            oppn_pawns_n = num_pieces_list_n[oppn_ind]
            oppn_kings_n = num_pieces_list_n[oppn_ind + 2]
            r_1 = agent_pawns - agent_pawns_n
            r_2 = agent_kings - agent_kings_n
            r_3 = oppn_pawns - oppn_pawns_n
            r_4 = oppn_kings - oppn_kings_n
            self.rewardX = r_3 * 0.2 + r_4 * 0.3 + r_1 * (-0.4) + r_2 * (-0.5)
             
            #Calculating Reward for agent 2
            agent_ind = 1
            oppn_ind = 0
            
            num_pieces_list = self.lastState.get_pieces_and_kings()
            agent_pawns = num_pieces_list[agent_ind]
            agent_kings = num_pieces_list[agent_ind + 2]
            oppn_pawns = num_pieces_list[oppn_ind]
            oppn_kings = num_pieces_list[oppn_ind + 2]

            num_pieces_list_n = self.currentState.get_pieces_and_kings()
            agent_pawns_n = num_pieces_list_n[agent_ind]
            agent_kings_n = num_pieces_list_n[agent_ind + 2]
            oppn_pawns_n = num_pieces_list_n[oppn_ind]
            oppn_kings_n = num_pieces_list_n[oppn_ind + 2]
            r_1 = agent_pawns - agent_pawns_n
            r_2 = agent_kings - agent_kings_n
            r_3 = oppn_pawns - oppn_pawns_n
            r_4 = oppn_kings - oppn_kings_n
            self.rewardO = r_3 * 0.2 + r_4 * 0.3 + r_1 * (-0.4) + r_2 * (-0.5)

        if self.rewardX == 0:
            self.rewardX = REWARD_DEFAULT
        if self.rewardO == 0:
            self.rewardO = REWARD_DEFAULT
                
    def get_last_rewardO(self):
        return self.rewardO
    
    def get_last_rewardX(self):
        return self.rewardX
    
    


    def get_actions(self,state=None):
        """
            Returns all applicable actions (non-marked fields)
        """
        if state is None:
            state = self.currentState
            
        return state.get_legal_actions()
                










