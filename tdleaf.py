from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import heapq as hq
import copy
import h5py

NEGINF = -9999999999999
POSINF = 999999999999

class evalFun:
    def __init__(self):
        #Always read in training from file
        weightsFile = open('weights.txt', 'r')
        rawIn = weightsFile.readline()
        self.weights = np.array(list(map(float, rawIn.split(','))))
        assert(len(self.weights) == 64)
        self.Lambda = 0.5
        self.Alpha = 0.1**10
        #set of positions in a game
        self.S = []
        #number of states
        self.N = 13
        #depth
        self.d = 5

    def setS(self, S):
        self.S = S

    def predict(self, position):
        return(np.dot(self.weights, position))

    #deritivative of inner product 
    #make sure to change if predict has changed
    def derivative(self, position):
        return position

    def train(self):
        lam = self.Lambda
        pos = self.S
        self.N = len(self.S)
        predicts = [None]*self.N
        for j in xrange(self.N):
            predicts[j] = self.predict(pos[j])
        for j in xrange(self.N-1):
            s = sum([self.Lambda**(k-j)*(predicts[k+1] - predicts[k]) for k in xrange(j, self.N-1)])
            self.weights += self.Alpha*s*self.derivative(pos[j])


        #After any training, save what we've done
        weightsFile = open('weights.txt', 'w')
        toWrite = ','.join(list(map(str,(self.weights))))
        weightsFile.write(toWrite)

def getGuess(model, position):
    x = np.array(position)
    x = x.reshape(1,65,1,1)
    out = model.predict(x)
    return out[0]

def getNBest(position, predictions, n, color):
    testPos = copy.deepcopy(position)
    testPos.pop()
    assert(len(testPos) == 64)

    withIndexes = [(predictions[i],i) for i in range(len(predictions))]
    valMoves = filter(lambda x: validateMove(testPos, x[1] % 8, x[1] // 8, color), withIndexes)

    largest = hq.nlargest(n, valMoves, key=(lambda x: x[0]))

    return list(map(lambda x: x[1], largest))

def up(x, y):
    return (x, y - 1)

def down(x, y):
    return (x, y + 1)

def left(x, y):
    return (x - 1, y)

def right(x, y):
    return (x + 1, y)

def upleft(x, y):
    return (x - 1, y - 1)

def upright(x, y):
    return (x + 1, y - 1)

def downleft(x, y):
    return (x - 1, y + 1)

def downright(x, y):
    return (x + 1, y + 1)

def pAt(board,x,y):
    return board[x + 8 * y]

def validateDir(board, x, y, dir, color):
    (x, y) = dir(x, y)
    if(x < 0 or x >= 8 or y < 0 or y >= 8 or pAt(board,x,y) != -color):
        return False
    while(x >= 0 and x < 8 and y >= 0 and y < 8 and pAt(board,x,y) != 0):
        if(pAt(board,x,y) == color):
            return True
        (x,y) = dir(x, y)
    return False

def flipDir(board, x, y, dir, color):
    (x, y) = dir(x, y)
    while(pAt(board,x,y) == -color):
        board[x + 8 * y] = color
        (x, y) = dir(x, y)

def validateMove(board, x, y, color):
    return pAt(board, x, y) == 0 and (
        validateDir(board, x, y, up, color) or
        validateDir(board, x, y, down, color) or
        validateDir(board, x, y, left, color) or
        validateDir(board, x, y, right, color) or
        validateDir(board, x, y, upleft, color) or
        validateDir(board, x, y, upright, color) or
        validateDir(board, x, y, downleft, color) or
        validateDir(board, x, y, downright, color))

def placeMove(board, x, y, color):
    board[x + 8 * y] = color
    if(validateDir(board, x, y, up, color)):
        flipDir(board, x, y, up, color)
    if(validateDir(board, x, y, down, color)):
        flipDir(board, x, y, down, color)
    if(validateDir(board, x, y, left, color)):
        flipDir(board, x, y, left, color)
    if(validateDir(board, x, y, right, color)):
        flipDir(board, x, y, right,  color)
    if(validateDir(board, x, y, upleft, color)):
        flipDir(board, x, y, upleft, color)
    if(validateDir(board, x, y, upright, color)):
        flipDir(board, x, y, upright, color)
    if(validateDir(board, x, y, downleft, color)):
        flipDir(board, x, y, downleft,  color)
    if(validateDir(board, x, y, downright, color)):
        flipDir(board, x, y, downright, color)

def isTerminal(position):
    for square in position:
        if square == 0: return False

    return True

def loadModel():
    model_name = 'model.h5'
    model_weights_name = 'model_weights.h5'

    model = load_model(model_name)
    model.load_weights(model_weights_name)

    return model

# Used for alpha-beta search. Can be thought of as "maxie"

def white(alpha, beta, model, evaluator, position, depth, nBest=5):
    if depth == 0 or isTerminal(position):
        return evaluator.predict(position)

    else:
        newPos = copy.deepcopy(position)
        newPos.append(sum(position))


        prob_prediction = getGuess(model, newPos)
        moves = getNBest(newPos, prob_prediction, nBest, 1)

        bestMoveVal = NEGINF
 
        for move in moves:
            nextPos = copy.deepcopy(position)
            placeMove(nextPos, move % 8, move // 8, 1)
            value = black(alpha, beta, model, evaluator, nextPos, depth-1)

            bestMoveVal = max(value, bestMoveVal)
            alpha = max(alpha, value)

            if beta <= alpha:
                break
        return bestMoveVal


# Used for alpha-beta search. Can be thought of as "minnie"
def black(alpha, beta, model, evaluator, position, depth, nBest=5):
    if depth == 0 or isTerminal(position):
        return evaluator.predict(position)

    else:
        newPos = copy.deepcopy(position)
        flipBoard(newPos)
        newPos.append(sum(position))

        prob_prediction = getGuess(model, newPos)
        moves = getNBest(newPos, prob_prediction, nBest, 1)

        bestMoveVal = POSINF

        for move in moves:
            nextPos = copy.deepcopy(position)
            placeMove(nextPos, move % 8, move // 8, -1)
            value = white(alpha, beta, model, evaluator, nextPos, depth-1)

            bestMoveVal = min(value, bestMoveVal)
            beta = min(beta, value)

            if beta <= alpha:
                break
        return bestMoveVal

def flipBoard(board):
    for i in range(len(board)):
        board[i]=-board[i]


if __name__ == '__main__':
    evaluator = evalFun()
    evaluator.train()
    model = loadModel()
