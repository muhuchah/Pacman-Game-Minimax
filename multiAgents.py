# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent
from pacman import GameState

mapp = np.zeros([20, 20], dtype = int)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """

    x, y = currentGameState.getPacmanPosition()
    
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)
        self.alpha = -INF
        self.beta = INF

INF = 1e9
class AIAgent(MultiAgentSearchAgent):
    def getAction(self, state: GameState):
        """
        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # TODO: Your code goes here
        # util.raiseNotDefined()

        # actions = gameState.getLegalActions(0)
        # print(actions)
        # action = random.randint(0, len(actions)-1)
        # return actions[action]

        x, y = state.getPacmanPosition()
        mapp[x, y] -= 1


        ghosts_number = state.getNumAgents() - 1;
        actions = state.getLegalActions(self.index)
        v = -INF
        for action in actions:
            new_v = self.min_value(state.generateSuccessor(self.index, action), self.depth - 1, self.alpha, self.beta, 1, ghosts_number)
            if new_v >= v:
                best_action = action
                v = new_v


        return best_action
            

    def max_value(self, state, depth, alpha, beta, ghosts_number):
        if state.isLose() or state.isWin() or depth == 0:
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(self.index)
        v = -INF
        for action in actions:
            g = state.generateSuccessor(self.index, action)
            v = max(v, self.min_value(g, depth-1, alpha, beta, 1, ghosts_number))
            if v >= beta:
                return v
            alpha = max(v, alpha)

        return v

    def min_value(self, state, depth, alpha, beta, index, ghosts_number):
        if state.isLose() or state.isWin() or depth == 0:
            return self.evaluationFunction(state)
        
        actions = state.getLegalActions(index)
        v = INF
        for action in actions:
            g = state.generateSuccessor(index, action)
            if index == ghosts_number:
                v = min(v, self.max_value(g, depth-1, alpha, beta, ghosts_number))
            else:
                v = min(v, self.min_value(g, depth-1, alpha, beta, index+1, ghosts_number))
            if v <= alpha:
                return v
            beta = min(v, beta)           
        return v
        
