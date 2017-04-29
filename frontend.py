from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import tdleaf as td

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
    bestVal = 0
    best = -1

    for i in range(len(predictions)):
        if(predictions[i] > bestVal and td.validateMove(position, i%8, i//8, color)):
            bestVal = predictions[i]
            best = i
    return best

def aiTurn(board,model,color):
    board.append(sum(board))
    board.append(color)
    move = getMove(model, board, color)
    board.pop()
    board.pop()
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


#Initializes and returns a game state based on user input
def startGame(model):
    print('Would you like to play first or second? (1 for first, 2 for second):')
    playerOrder = input() == "1"
    gameTurn(model, startVector, -1, playerOrder, True)


if __name__ == '__main__':
    model = td.loadModel()
    startGame(model)
