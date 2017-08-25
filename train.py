import os,subprocess,glob,sys
from PIL import Image
from math import exp,pi,sin,cos
import numpy as np

X = np.zeros([150,240,240,5],'float32')
with open("./demo/X.txt", "r") as f:
    num=-1
    row=-1
    column=-1
    ch=0
    for line in f:
        if (line.find('flag0')):
            num += 1
        elif (line.find('flag1')):
            row += 1
        elif (line.find('flag2')):
            column += 1
            ch = 0
        else:
            X[num,row,colomn,ch] = float(line)
            ch += 1
        
Xte = np.zeros([26,240,240,5],'float32')
with open("./demo/Xte.txt", "r") as f:
    num=-1
    row=-1
    column=-1
    ch=0
    for line in f:
        if (line.find('flag0')):
            num += 1
        elif (line.find('flag1')):
            row += 1
        elif (line.find('flag2')):
            column += 1
            ch = 0
        else:
            Xte[num,row,colomn,ch] = float(line)
            ch += 1

def mask_check(mask_path):
    mask = Image.open(mask_path)
    _x = int(mask.size[0]/4)
    _y = int(mask.size[1]/4)
    Y = np.zeros([1, _y*_x])
    mask = np.array(mask.resize((_x,_y)))
    for y in range(_y):
        for x in range(_x):
            if(mask[y,x]):
                Y[0,_x*y+x] = 1
    return Y

train_ans_path = glob.glob("C:/Users/shgtkmt/workspace/fourier_learning/train_ans/*.tif")
data_num = (len(train_ans_path) - 1) - 26
Y = mask_check(train_ans_path[data_num])
for i in range(data_num):
    Y = np.append(Y,mask_check(train_ans_path[i]),axis=0)

test_ans_path = glob.glob("C:/Users/shgtkmt/workspace/fourier_learning/train_ans/*.tif")
data_num = len(test_ans_path)-1
Yte = mask_check(test_ans_path[data_num])
for i in range(data_num - 150):
    Yte = np.append(Yte,mask_check(test_ans_path[i + 150]),axis=0)

from keras.models import Sequential
model = Sequential()
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

model.add(Conv2D(10, (3, 3), padding='same', input_shape=(240, 240, 5)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(20, (5, 5), padding='same',))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

'''
model.add(Conv2D(30, (4, 4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D((4, 4)))
model.add(Conv2D(10, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D((3, 3)))
'''

model.add(Conv2D(1, (3, 3), padding='same'))
model.add(Flatten())
#model.add(Dense(5000))
#model.add(LeakyReLU(0.2))
#model.add(Dropout(0.3))
model.add(Dense(60*60))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", 
              optimizer='adam',
             metrics=['accuracy'])

model.fit(X, Y, nb_epoch=10, batch_size=10)
loss_and_metrics = model.evaluate(Xte,Yte)

print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))
