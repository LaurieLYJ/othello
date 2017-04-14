#Created by Eliot Robson
#AID: erobson

#Made for my 10-401 machine learning project. Reads in data outputted by Ntest
#Othello AI and writes it to a file. File format is as follows:

#One line of the output file is a vector containing the following information.
# 1) The first 64 entries denote the othello board at a given time. The board is
#    flattened, so it represents each row of the game board, appeneded together,
#    from left to right. 1 denotes a white tile, 0 an empty space, and -1 a black tile
# 2) The next entry is the current game's score. The score is the number of white
#    tiles minus the number of black tiles. Thus, a negative score represents black
#    as winning, while a positive score represents white as winning
# 3) The final number represents the player whose turn it is to move. -1 means
#    it is black's turn, and 1 means it is white's turn
#The line after each input vector line, in (row, col) format, is
#the move that was selected by the AI from the game position in the
#previous line.

#The output file lines alternate in this pattern until the end of the file

#The output file is made such that the neural network to play the game will take
#in the input vector and be trained on the output data  on the line following
#an input vector.

#Recursively flattens a list
#From my own 112 homework 9
def flatten(L,spot=0):
    if(type(L) != list): return L #Single element means return itself
    elif(spot >= len(L)): return [] #Otherwise, return nothing to append

    else:
        element = flatten(L[spot], spot=0)
        rest = flatten(L,spot+1) #Look ma, no loops!

        if(type(element) == list): return element + rest
        else: return [element] + rest

#Reads in a board from a file, returns the board as a 2D-list
#Left in format as strings
def readBoard(trainingFile):
    boardLines = 8
    board = []

    for i in range(boardLines):
        rawLine = trainingFile.readline()

        if rawLine == "":
            print("Board not found!")
            return []

        line = rawLine.split()
        boardLine = line[1:8]
        board.append(boardLine)

    return board

#Reads in who the turn player is, assumes we already have board read in
#Returns 0 if player is white, 1 if player is black
def readPlayer(trainingFile):
    trainingFile.readline() #Read in bottom of board
    rawPlayer = trainingFile.readline()

    if rawPlayer == "":
        print("Player not found!")
        return -2

    elif rawPlayer == "Black to move\n": return -1
    else: return 1 #White player is moving

#Takes in a letter for a row, converts it to an int
def colToInt(colLetter):
    if colLetter == "A": return 1
    elif colLetter == "B": return 2
    elif colLetter == "C": return 3
    elif colLetter == "D": return 4
    elif colLetter == "E": return 5
    elif colLetter == "F": return 6
    elif colLetter == "G": return 7
    elif colLetter == "H": return 8
    else:
        print("Invalid column as input!")
        return -1

#After turn player is read in, skip to move AI decided should be played
#Reads in move, and returns the move in (row,column) format as a tuple
def readMove(trainingFile):
    rawLine = trainingFile.readline()
    while(rawLine[0] != "=" and rawLine != ""):
        rawLine = trainingFile.readline()

    if rawLine == "":
        print("Move not found!!")
        return (-1,-1)

    else:
        if rawLine[4] == "p":
            return (0,0) #Means player passed

        col = colToInt(rawLine[4])
        row = int(rawLine[5])

        return (row,col)

#Reads from target file until we reach a board
#EOF reached true if board found, false if EOF reached
def readUntilBoard(trainingFile):
    boardHeader = "   A B C D E F G H  \n"
    newLine = ""
    while(newLine != boardHeader):
        newLine = trainingFile.readline()
        if newLine == "" or newLine[0:16] == "status Learning":
            return False

    return True

#From raw inputs, makes vector to output to file
#Format of returned vector detailed at top of file
def makeVector(rawBoard, rawPlayer, rawMove):
    boardVector = flatten(rawBoard)
    outVector = []
    score = 0

    for space in boardVector:
        #0 is empty space, 1 is white, -1 is black
        #O = white piece, * = black piece, - = empty space
        nextVect = "0"
        if space == 'O':
            nextVect = "1"
            score += 1
        elif space == "*":
            nextVect = "-1"
            score -= 1
        elif space == "-":
            nextVect = "0"
        else:
            print("Invalid board piece for conversion!")

        outVector.append(nextVect)

    outVector.append(str(score))
    outVector.append(str(rawPlayer))
    outVector.append("\n")

    outVector.append(str(rawMove[0]))
    outVector.append(str(rawMove[1]))
    outVector.append("\n")

    return " ".join(outVector)

#Skips analysis in training file where computer goes over game
def skipAnalysis(trainingFile):
    fileLine = trainingFile.readLine()

    while(fileLine[0:18] != "Analysis complete"):
        fileLine = trainingFile.readLine()

        if fileLine == "":
            print("Reached EOF while skipping game analysis!")
            return False

    return True


def main():
    #@TODO Change directory based on computer
    directory = "C:\\Users\\Eliot\\Documents\\CMU\\S17\\10-401\\Othello Project\\othello\\training data converter\\"

    rawFolder = "Raw Output\\"
    trainingFileName = "games_2.txt"
    trainingFile = open(directory + rawFolder + trainingFileName, 'r')

    outFileName = "converted_"
    outputFolder = "Converted Files\\"
    outFile = open(directory + outputFolder + outFileName + trainingFileName, 'w')

    keepRunning = True

    while(keepRunning):
        while(readUntilBoard(trainingFile)):

            #Read in board, player, and computer's move
            rawBoard = readBoard(trainingFile)
            rawPlayer = readPlayer(trainingFile)
            rawMove = readMove(trainingFile)

            #If we se an error, return
            if rawMove[1] == -1 or rawBoard == [] or rawPlayer == -2:
                print("File, move, or board not found!!")
                return None

            #Write training vector to file
            trainingVector = makeVector(rawBoard, rawPlayer, rawMove)
            outFile.write(trainingVector)

        #Updates running label
        keepRunning = skipAnalysis(trainingFile)

main()
