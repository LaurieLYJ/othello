from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import tdleaf as td
import copy

NEGINF = -9999999999999
POSINF = 999999999999

#Board vector without the starting player, to be initialized
startVector = [0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 1,-1, 0, 0, 0,
               0, 0, 0,-1, 1, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0]

def getMove(model, position):
    position.append(sum(position))
    predictions = td.getGuess(model, position)
    possBests = td.getNBest(position, predictions, 7, 1)
    evaluator = td.evalFun()
    depth = 7
    bestMove = -1
    bestMoveVal = 0
    position.pop()

    bestMoveVal = NEGINF
    for move in possBests:
        nextPos = copy.deepcopy(position)
        td.placeMove(nextPos, move % 8, move // 8, 1)
        value = td.black(NEGINF, POSINF, model, evaluator, nextPos, depth-1)

        if value > bestMoveVal:
            bestMove = move
            bestMoveVal = value

    return bestMove

def aiTurn(board,model,color):
    if(color == -1):
        td.flipBoard(board)
    move = getMove(model, board)
    if(color == -1):
        td.flipBoard(board)
    if(move != -1):
        td.placeMove(board, move % 8, move // 8, color)
    return move != -1

def playerMove(board,color):
    validMove = False
    while(not validMove):
        print("Where do you want to move? (Must follow pattern of \"A1\")")
        playerMove = input()
        moveX = ord(playerMove[0]) - ord('A')
        moveY = int(playerMove[1]) - 1
        validMove = td.validateMove(board,moveX,moveY,color)
        if(not validMove):
            print("That move is not valid")
    print("{:d},{:d}".format(moveX, moveY))
    td.placeMove(board, moveX, moveY, color)

def playerTurn(board,color):
    canMove = printBoard(board,color)
    if(canMove):
        playerMove(board,color)
        return True
    else:
        print("No valid moves, hit enter to continue")
        input()
        return False

#Takes in a boardVector and prints the game state according to a given vector
def printBoard(boardVector,color):
    rows = [boardVector[x:x+8] for x in range(0, 64, 8)]
    lettersRow = '  A B C D E F G H\n'
    assert(len(boardVector) == 64)

    canMove = False
    output = lettersRow
    for i in range(len(rows)):
        row = str(i+1) + ' '
        for j in range(len(rows[i])):
            if(rows[i][j] == 1):
                row += 'O '
            elif(rows[i][j] == -1):
                row += '* '
            elif(td.validateMove(boardVector, j, i, color)):
                canMove = True
                row += '- '
            else:
                row += '  '
        row += (str(i+1) + '\n')
        output += row

    output += lettersRow
    print(output)

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
        elif(sum(board[0:65])<0):
            print("WHITE WINS!")
        else:
            print("BLACK WINS!")
    else:
        gameTurn(model, board, -color, not player, madeMove)


#Initializes and returns a game state based on user input
def startGame(model):
    print('Would you like to play first or second? (1 for first, 2 for second):')
    playerOrder = input() == "1"
    gameTurn(model, startVector, -1, playerOrder, True)


if __name__ == '__main__':
    model = td.loadModel()
    startGame(model)
