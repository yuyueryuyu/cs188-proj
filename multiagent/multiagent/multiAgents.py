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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newCapsule = successorGameState.getCapsules()
        gdis = 99999
        fdis = 99999
        cdis = 99999
        for i, ghost in enumerate(newGhostStates) :
            if newScaredTimes[i] == 0:
                gdis = min(gdis, manhattanDistance(newPos,ghost.getPosition())) 
            else:
                gdis = min(gdis, newScaredTimes[i]-manhattanDistance(newPos,ghost.getPosition())) 
        for food in newFood.asList():
            fdis = min(fdis,manhattanDistance(newPos,food))
        if len(currentGameState.getFood().asList()) != len(newFood.asList()):
            fdis = 0
        for cap in newCapsule:
            cdis = min(cdis,manhattanDistance(newPos,cap))
        if len(currentGameState.getCapsules()) != len(newCapsule):
            cdis = -99999
        if gdis>6:
            gdis=100
        return successorGameState.getScore() +gdis - fdis - cdis

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

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
        "*** YOUR CODE HERE ***"
        def MiniMax(dep,nowState):
            ret = []
            agentNum = nowState.getNumAgents()
            for action in nowState.getLegalActions(dep%agentNum):
                newState = nowState.generateSuccessor(dep%agentNum, action)
                if newState.isWin() or newState.isLose() or dep == self.depth * agentNum - 1:
                    ret.append((self.evaluationFunction(newState),action))
                else:
                    retVal = MiniMax(dep+1,newState)
                    if (dep+1)%agentNum == 0:
                        ret.append((max(retVal)[0],action))
                    else:
                        ret.append((min(retVal)[0],action))
            return ret
        retVal = MiniMax(0,gameState)
        mn = -200000
        mact = None
        for val,act in retVal:
            if val > mn:
                mn = val
                mact = act
        return mact
                
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxVal(dep,nowState,al,bt):
            v = float('-inf'),""
            agentNum = nowState.getNumAgents()
            for action in nowState.getLegalActions(dep%agentNum):
                newState = nowState.generateSuccessor(dep%agentNum, action)
                if newState.isWin() or newState.isLose() or dep == self.depth * agentNum - 1:
                    nVal = self.evaluationFunction(newState),action
                else:
                    if (dep+1)%agentNum == 0:
                        nVal = maxVal(dep+1,newState,al,bt)
                    else:
                        nVal = minVal(dep+1,newState,al,bt)
                v = max(v,(nVal[0],action))
                
                if v[0] > bt:
                    return v
                al = max(al,v[0])
            return v
        def minVal(dep,nowState,al,bt):
            v = float('inf'),""
            agentNum = nowState.getNumAgents()
            for action in nowState.getLegalActions(dep%agentNum):
                newState = nowState.generateSuccessor(dep%agentNum, action)
                if newState.isWin() or newState.isLose() or dep == self.depth * agentNum - 1:
                    nVal = self.evaluationFunction(newState),action
                else:
                    if (dep+1)%agentNum == 0:
                        nVal = maxVal(dep+1,newState,al,bt)
                    else:
                        nVal = minVal(dep+1,newState,al,bt)
                v = min(v,(nVal[0],action))
                if v[0] < al:
                    return v 
                bt = min(bt,v[0])
            return v
        retVal = maxVal(0,gameState,float('-inf'),float('inf'))
        return retVal[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def ExpectiMax(dep,nowState):
            ret = []
            agentNum = nowState.getNumAgents()
            for action in nowState.getLegalActions(dep%agentNum):
                newState = nowState.generateSuccessor(dep%agentNum, action)
                if newState.isWin() or newState.isLose() or dep == self.depth * agentNum - 1:
                    ret.append((self.evaluationFunction(newState),action))
                else:
                    retVal = ExpectiMax(dep+1,newState)
                    if (dep+1)%agentNum == 0:
                        ret.append((max(retVal)[0],action))
                    else:
                        avg = 0
                        for val in retVal:
                            avg += val[0]
                        avg /= len(retVal)
                        ret.append((avg,action))
            return ret
        retVal = ExpectiMax(0,gameState)
        mn = -200000
        mact = None
        for val,act in retVal:
            if val > mn:
                mn = val
                mact = act
        return mact
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Similar to Q1, but I changed something.
    The evaluation of each states 
    =   the score of the state 
        + when ghosts are scared:           the minimum of (the scared time of ghosts - the manhattan distance to ghosts)
          or when ghosts are not scared:    the minimum manhattan distance to ghosts 
        - the maximum manhattan distance to food and capsules
    gdis: the scared time of ghosts - the manhattan distance to ghosts (when ghosts are scared)
          the minimum manhattan distance to ghosts (when ghosts are not scared)
    =>  This ensures that when ghosts are scared, Pacman should prioritize hunting them and eating food, rather than avoiding them. 
        However, if the remaining scared time is too short for hunting or the ghosts are not scared, Pacman should still avoid them.
        Also, if the ghosts are far away or the remaining scared time is long (if gdis is "large" -- I simply set to be greater than 6), 
        Pacman can safely focus on eating food without worrying about the ghosts. 
        In such cases, I add a bonus of 100 to the score for these states, so Pacman can choose to stay in these states
    fdis, cdis: the maximum manhattan distance to food and capsules
    =>  Pacman should move towards food and capsules and eat them as soon as possible. Therefore, I subtract fdis and cdis from the score to encourage this behavior.
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]
    currCapsules = currentGameState.getCapsules()
    gdis = 99999
    fdis = 99999
    cdis = 99999
    for i, ghost in enumerate(currGhostStates) :
        if currScaredTimes[i] == 0:
            gdis = min(gdis, manhattanDistance(currPos,ghost.getPosition())) 
        else:
            gdis = min(gdis, currScaredTimes[i]-manhattanDistance(currPos,ghost.getPosition())) 
    for food in currFood.asList():
        fdis = max(fdis,manhattanDistance(currPos,food))
    for cap in currCapsules:
        cdis = max(cdis,manhattanDistance(currPos,cap))
    if gdis>6:
        gdis+=100
    return currentGameState.getScore() + gdis - fdis - cdis
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
