from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import tdleaf as td
import copy
import frontend as fd
import random

dataFile = "training data converter/final_output.txt"
with open(dataFile) as f:
	content = f.readlines()
content = [l.strip() for l in content]
del content[1::2]
content = [list(map(float, content[i].split())) for i in xrange(len(content))]
for i in xrange(len(content)):
	content[i].pop()
	content[i].pop()
	assert(len(content[i]) == 64)
content = random.sample(content, 100)
#return a list of 12 moves played by AI, given a startingPos
def makePositions(model, startingPos):
	l = []
	l.append(np.array(copy.deepcopy(startingPos)))
	turns = [-1, 1]
	color = turns[random.randint(0, 1)]
	board = copy.deepcopy(startingPos)
	for i in xrange(12):
		valid = fd.aiTurn(board, model, color)
		if not valid:
			break 
		l.append(np.array(copy.deepcopy(board)))
		if color == -1:
			color = 1
		else:
			color = -1
	return l 


def main():
	evaluator = td.evalFun()
	model = td.loadModel()
	for i in xrange(len(content)):
		evaluator.setS(makePositions(model, content[i]))
		evaluator.train()
		print "case {} done.".format(i)


main()