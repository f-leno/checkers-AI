"""
    Expert agent for Tic Tac Toe
    Adapted from:
    
    https://github.com/robbiebarrat/unbeatable_tictactoe/blob/master/TicTacToe.py
    https://en.wikipedia.org/wiki/Tic-tac-toe
    https://mblogscode.wordpress.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/
    Author: Felipe Leno (f.leno@usp.br)

"""
import math
from agents.agent import Agent

from randomSeeds.randomseeds import Seeds



class ExpertTicTacToeAgent(Agent):
    aiturn = None
    rnd = None
    def __init__(self):
        self.rnd = Seeds().EXPERT_AGENT_SEED
        self.aiturn = 0
    def transform_action(self,act):
        self.aiturn += 1
        y = int(math.floor(act / 3))
        x = int(math.floor(act % 3))
        #print((x,y))
        return (x,y)
        
    def observe_reward(self,state,action,statePrime,reward):
        if reward != 0:
            self.aiturn = 0
    def select_action(self, state):
        """
            Adapted from the original AI, see comments there
            
        """
        #print(state)
        opponentMark = "X" if self.marker == "O" else "O"
        return self.transform_action(self.getComputerMove(state, self.marker, opponentMark, self.environment.board.empty))
        completes = ([0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6])
        
        corners = [0, 2, 6, 8]
    def getComputerMove(self,state,mark,opponentMark,empty):
        def checkWin(state,empty):
            completes = ([0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6])
            for posics in completes:
                if state[posics[0]] != empty:
                    if state[posics[0]] == state[posics[1]] and state[posics[1]] == state[posics[2]]:
                        return True  
            return False
        def testWinMove(state, mark,empty, i):
            # b = the board
            # mark = 0 or X
            # i = the square to check if makes a win 
            state = list(state)
            state[i] = mark
            state = tuple(state)
       
            return checkWin(state,empty)
        
        def testForkMove(state, mark, empty, i):
            # Determines if a move opens up a fork
            state = list(state)
            state[i] = mark
            state = tuple(state)
            winningMoves = 0
            for j in range(0, 9):
                if state[j] == empty and testWinMove(state, mark,empty, j):
                    winningMoves += 1
            return winningMoves >= 2
        def fork(state,opponentMark,mark,empty):
            exists = False
            forkMove = None
            # Check agent fork opportunities
            for i in range(0, 9):
                if state[i] == empty and testForkMove(state, mark, empty, i):
                    exists = True
                    forkMove = i
                    break
            #  Check block fork opportunities
            if not exists:
                forkNumber = 0
                forkMoves = []
                for i in range(0, 9):
                    if state[i] == empty and testForkMove(state, opponentMark,empty, i):
                        forkNumber += 1
                        forkMoves.append(i)
                        exists = True
                if forkNumber == 1:
                    forkMove = forkMoves[0]
                if forkNumber >= 2:
                    #Check if, when defending one of the for movies, the agent will force
                    #the opponent to defend himself
                    for fm in forkMoves:
                        #Only execute for if no appropriate position was found
                        if forkMove is not None:
                            break
                        stateCheck = list(state)
                        stateCheck[fm] = mark
                        stateCheck = tuple(stateCheck)
                        for j in range(0, 9):
                            if stateCheck[j] == empty and testWinMove(stateCheck, mark,empty, j):
                                if not testForkMove(stateCheck, opponentMark,empty, j):
                                    forkMove = fm
                                    break
                         
                    
                    if forkMove is None:
                        for j in [1, 3, 5, 7]:
                            if state[j]==empty:
                                forkMove =  j
                        if forkMove is None:
                            forkMove = rnd.choice(forkMoves)

        
            return exists,forkMove
        # Check computer win moves
        for i in range(0, 9):
            if state[i] == empty and testWinMove(state, mark,empty, i):
                return i
        # Check player win moves
        for i in range(0, 9):
            if state[i] == empty and testWinMove(state, opponentMark, empty, i):
                return i
        # Check computer fork opportunities
        for i in range(0, 9):
            if state[i] == empty:
                exists, forkMove = fork(state,opponentMark,mark,empty)
                if exists:
                    return forkMove
        # Play center
        if state[4] == empty:
            return 4
        # Play a corner
        for i in [0, 2, 6, 8]:
            if state[i] == empty:
                return i
        #Play a side
        for i in [1, 3, 5, 7]:
            if state[i] == empty:
                return i
        
        
        # Pretty self-explanatory, just chooses a random empty corner.
#         def cornerchoice(corners, state):
#             goodchoices = []
#             if not alreadymoved:
#                 for i in corners:
#                     if state[i] == self.environment.board.empty:
#                         goodchoices.append(i)
#                 if len(goodchoices) > 0:
#                     return self.transform_action(self.rnd.choice(goodchoices))
#                 else:
#                     for i in range(9):
#                         if state[i] == self.environment.board.empty:
#                             return self.transform_action(i)
        
#         def checkWin(state,empty):
#             completes = ([0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6])
#             for posics in completes:
#                 if state[posics[0]] != empty:
#                     if state[posics[0]] == state[posics[1]] and state[posics[1]] == state[posics[2]]:
#                         return True  
#             return False
#         def testWinMove(state, mark,empty, i):
#             # b = the board
#             # mark = 0 or X
#             # i = the square to check if makes a win 
#             state = list(state)
#             state[i] = mark
#             state = tuple(state)
#        
#             return checkWin(state,empty)
#         
#         def testForkMove(state, mark, empty, i):
#             # Determines if a move opens up a fork
#             state = list(state)
#             state[i] = mark
#             state = tuple(state)
#             winningMoves = 0
#             for j in range(0, 9):
#                 if state[j] == empty and testWinMove(state, mark,empty, j):
#                     winningMoves += 1
#             return winningMoves >= 2
#         def fork(state,opponentMark,mark,empty):
#             exists = False
#             forkMove = None
#             # Check agent fork opportunities
#             for i in range(0, 9):
#                 if state[i] == empty and testForkMove(state, mark, empty, i):
#                     exists = True
#                     forkMove = i
#                     break
#             #  Check block fork opportunities
#             if not exists:
#                 for i in range(0, 9):
#                     if state[i] == empty and testForkMove(state, opponentMark,empty, i):
#                         exists = True
#                         forkMove = i
#                         break
#         
#             return exists,forkMove
#         
#         
#         
#         # Checks to see if there are any possible ways for the game to end next turn, and takes proper action.
#         for x in completes:
#             # Offensive
#             if state[x[0]] == self.marker and state[x[1]] == self.marker and state[x[2]] == self.environment.board.empty:
#                 return self.transform_action(x[2])
#                 alreadymoved = True
#                 break
#             if state[x[1]] == self.marker and state[x[2]] == self.marker and state[x[0]] == self.environment.board.empty:
#                 return self.transform_action(x[0])
#                 alreadymoved = True
#                 break
#             if state[x[0]] == self.marker and state[x[2]] == self.marker and state[x[1]] == self.environment.board.empty:
#                 return self.transform_action(x[1]) 
#                 alreadymoved = True
#                 break
#         # Tweaked it here a little bit, thanks to reddit user mdond for letting me know. It defending items closer to the
#         # start of the list 'pairs' before it would play offensive with items later in 'pairs'.
#            
#         for x in completes:
#             if alreadymoved == False:
#                 # Defensive
#                 if state[x[0]] == opponentMark and state[x[1]] == opponentMark and state[x[2]] == self.marker:
#                     return self.transform_action(x[2])
#                     alreadymoved = True
#                     break
#                 if state[x[1]] == opponentMark and state[x[2]] == opponentMark and state[x[0]] == self.marker:
#                     return self.transform_action(x[0])
#                     alreadymoved = True
#                     break
#                 if state[x[0]] == opponentMark and state[x[2]] == opponentMark and state[x[1]] == self.marker:
#                     return self.transform_action(x[1]) 
#                     alreadymoved = True
#                     break
#                    
#         #Create a fork or block fork
#         exists,forkMove = fork(state,opponentMark,self.marker, self.environment.board.empty)
#         if exists:
#             return self.transform_action(forkMove)
#         
#                   
#         #take center if not taken
#         if state[4] == self.environment.board.empty:
#                 return self.transform_action(4)
#                 alreadymoved = True   
#         
#                 #Take Opposite corners
#         if state[0] == opponentMark and state[8] == self.environment.board.empty:
#             return self.transform_action(8)
#         if state[2] == opponentMark and state[6] == self.environment.board.empty:
#             return   self.transform_action(6)
#         if state[6] == opponentMark and state[2] == self.environment.board.empty:
#             return   self.transform_action(2)
#         if state[8] == opponentMark and state[0] == self.environment.board.empty:
#             return   self.transform_action(0)
#         
#         # Any free corner or free space, if needed
#         return cornerchoice(corners, state)
#    



