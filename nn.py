import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.callbacks import ModelCheckpoint

batch_size = 100
epochs = 50

#map each location to a category
groups = [[0]*9 for i in xrange(9)]
for i in xrange(1, 9):
	for j in xrange(1, 9):
		groups[i][j] = (i-1)*8+(j-1)

X_trn = []
Y_trn = []

#read from the training data
dataFile = "/Users/liuyujin/othello/training data converter/training_data_1.txt"
with open(dataFile) as f:
	content = f.readlines()
content = [l.strip() for l in content]
for i in xrange(len(content)):
	if i % 2 == 0:
		#features 
		X_trn.append(map(float, content[i].split()))
	else:
		#class 
		l = map(int, content[i].split())
		Y_trn.append(groups[l[0]][l[1]])
X_trn = np.array(X_trn)
Y_trn = np.array(Y_trn)

input_cols = 58

num_groups = 64

input_shape = (input_cols, 1, 1)

convolution_layers = [
    Convolution2D(16, 3, 1, activation='relu', input_shape=input_shape),
    Convolution2D(32, 3, 1, activation='relu'),
    MaxPooling2D(pool_size=(2, 1))
]

classification_layers = [
    Flatten(),
    Dense(58*4, activation='relu'),
    Dense(num_groups, activation='softmax')
]

model = Sequential(convolution_layers + classification_layers)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_trn = X_trn.reshape(X_trn.shape[0], input_cols,1, 1)
Y_trn = np_utils.to_categorical(Y_trn, num_groups)
model.fit(X_trn, Y_trn, batch_size=batch_size, nb_epoch=epochs)

