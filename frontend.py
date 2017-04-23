from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint

testVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 1, -1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]

#Takes in a boardVector and prints the game state according to a given vector
def printBoard(boardVector):
    rows = [boardVector[x:x+8] for x in range(0, 56, 7)]
    lettersRow = '  A B C D E F G H\n'

    output = lettersRow
    for i in range(len(rows)):
        row = str(i+1) + ' '
        for square in rows[i]:
            if(square == 1):
                row += 'O '
            elif(square == -1):
                row += '* '
            else:
                row += '- '
        row += (str(i+1) + '\n')
        output += row

    output += lettersRow

    playerIndex = len(boardVector)-1
    player = boardVector[playerIndex]

    if(player == 1):
        output += 'White to move'
    else:
        output += 'Black to move'

    print(output)

def loadModel():
    model_name = 'model.h5'
    model_weights_name = 'model_weights.h5'

    model = load_model(model_name)
    model.load_weights(model_weights_name)

    return model

if __name__ == '__main__':
    model = loadModel()
    printBoard(testVector)
