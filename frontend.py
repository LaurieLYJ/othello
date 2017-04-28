from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np

#Board vector without the starting player, to be initialized
startVector = [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1,-1, 0, 0, 0,
               0, 0, 0,-1, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0]

def getMove(model, position, color):
    x = np.array(position)
    x = x.reshape(1,66,1,1)
    out = model.predict(x)
    return findBest(position, out[0], color)

def findBest(position, predictions, color):
    max = 0
    best = -1
    for i in range(len(predictions)):
        if(predictions[i]>max and validateMove(position, i, color)):
            max = predictions[i]
            best = i
    return best

def aiTurn(board,model,color):
    board.append(sum(board))
    board.append(color)
    move = getMove(model, board, color)
    board.pop()
    board.pop()
    if(move != -1):
        placeMove(board, move, color)
    return move != -1

def playerMove(board,color):
    validMove = False
    while(not validMove):
        print("Where do you want to move? (Must follow pattern of \"A1\")")
        playerMove = input()
        move = ord(playerMove[0])-ord('A') + 8*(int(playerMove[1]) -1)
        validMove = validateMove(board,move,color)
        if(not validMove):
            print("That move is not valid")
    print(move)
    placeMove(board, move, color)

def playerTurn(board,color):
    canMove = printBoard(board,color)
    if(canMove):
        playerMove(board,color)
        return True
    else:
        print("No valid moves, hit enter to continue")
        raw_input()
        return False


def up(i):
    return i+8

def down(i):
    return i-8

def left(i):
    return i-1

def right(i):
    return i+1

def upleft(i):
    return i-9

def upright(i):
    return i-7

def downleft(i):
    return i+7

def downright(i):
    return i+9

def validateDir(board, i, dir, color):
    i = dir(i)
    if(i < 0 or i >=64 or board[i] != -color):
        return False
    while(i>=0 and i < 64 and board[i] != 0):
        if(board[i] == color):
            return True
        i = dir(i)
    return False

def flipDir(board, i, dir, color):
    i = dir(i)
    while(board[i] == -color):
        board[i] = color
        i = dir(i)

def validateMove(board, i, color):
    return board[i] == 0 and (
        validateDir(board, i, up, color) or 
        validateDir(board, i, down, color) or 
        validateDir(board, i, left, color) or 
        validateDir(board, i, right, color) or 
        validateDir(board, i, upleft, color) or 
        validateDir(board, i, upright, color) or 
        validateDir(board, i, downleft, color) or 
        validateDir(board, i, downright, color))

def placeMove(board, i, color):
    board[i] = color
    if(validateDir(board, i, up, color)):
        flipDir(board, i, up, color)
    if(validateDir(board, i, down, color)):
        flipDir(board, i, down, color)
    if(validateDir(board, i, left, color)):
        flipDir(board, i, left, color)
    if(validateDir(board, i, right, color)):
        flipDir(board, i, right,  color)
    if(validateDir(board, i, upleft, color)):
        flipDir(board, i, upleft, color)
    if(validateDir(board, i, upright, color)):
        flipDir(board, i, upright, color)
    if(validateDir(board, i, downleft, color)):
        flipDir(board, i, downleft,  color)
    if(validateDir(board, i, downright, color)):
        flipDir(board, i, downright, color)
    

#Takes in a boardVector and prints the game state according to a given vector
def printBoard(boardVector,color):
    rows = [boardVector[x:x+8] for x in range(0, 64, 8)]
    lettersRow = '  A B C D E F G H\n'

    canMove = False
    output = lettersRow
    for i in range(len(rows)):
        row = str(i+1) + ' '
        for j in range(len(rows[i])):
            if(rows[i][j] == 1):
                row += 'O '
            elif(rows[i][j] == -1):
                row += '* '
            elif(validateMove(boardVector, 8*i + j, color)):
                canMove = True
                row += '- '
            else:
                row += '  '
        row += (str(i+1) + '\n')
        output += row

    output += lettersRow

    print(output)
    return canMove

def gameTurn(model, board, color, player, prev):
    if(player):
        madeMove = playerTurn(board, color)
    else:
        madeMove = aiTurn(board, model, color)
    if((not madeMove) and(not prev)):
        if(sum(board) == 0):
            print("TIE!")
        elif(sum(board)>0):
            print("WHITE WINS!")
        else:
            print("BLACK WINS!")
    else:
        gameTurn(model, board, -color, not player, madeMove)
           


def loadModel():
    model_name = 'model.h5'
    model_weights_name = 'model_weights.h5'

    model = load_model(model_name)
    model.load_weights(model_weights_name)

    return model

#Initializes and returns a game state based on user input
def startGame(model):
    print('Would you like to play first or second? (1 for first, 2 for second):')
    playerOrder = input() == "1"
    gameTurn(model, startVector, -1, playerOrder, True)
    

if __name__ == '__main__':
    model = loadModel()
    startGame(model)
