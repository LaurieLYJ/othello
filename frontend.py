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
        if(predictions[i]>max and validateMove(position, i%8, i/8, color)):
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
        placeMove(board, move % 8, move / 8, color)
    return move != -1

def playerMove(board,color):
    validMove = False
    while(not validMove):
        print("Where do you want to move? (Must follow pattern of \"A1\")")
        playerMove = input()
        moveX = ord(playerMove[0]) - ord('A')
        moveY = int(playerMove[1]) - 1
        validMove = validateMove(board,moveX,moveY,color)
        if(not validMove):
            print("That move is not valid")
    print("{:d},{:d}".format(moveX, moveY))
    placeMove(board, moveX, moveY, color)

def playerTurn(board,color):
    canMove = printBoard(board,color)
    if(canMove):
        playerMove(board,color)
        return True
    else:
        print("No valid moves, hit enter to continue")
        raw_input()
        return False


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
            elif(validateMove(boardVector, j, i, color)):
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
