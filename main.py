import os,subprocess,glob,sys
from PIL import Image
from math import exp,pi,sin,cos
import numpy as np



X = np.zeros([150,240,240,5],'float32')
with open("./../X.txt", "r") as f:
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
with open("./../Xte.txt", "r") as f:
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
    _x = int(mask.size[0])
    _y = int(mask.size[1])
    Y = np.zeros([1, _y, _x])
    mask = np.array(mask)
    for y in range(_y):
        for x in range(_x):
            if(mask[y,x]):
                Y[0,y,x] = 1
    return Y

train_ans_path = glob.glob("C:/Users/shgtkmt/workspace/cnn_fourier/train_ans/*.tif")
data_num = (len(train_ans_path) - 1) - 26
Y = mask_check(train_ans_path[data_num])
for i in range(data_num):
    Y = np.append(Y,mask_check(train_ans_path[i]),axis=0)

test_ans_path = glob.glob("C:/Users/shgtkmt/workspace/cnn_fourier/train_ans/*.tif")
data_num = len(test_ans_path)-1
Yte = mask_check(test_ans_path[data_num])
for i in range(data_num - 150):
    Yte = np.append(Yte,mask_check(test_ans_path[i + 150]),axis=0)



from keras.models import Sequential, Model
#model = Sequential()
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

input_img = Input(shape=(240,240,5))
x = Conv2D(10, (3, 3), padding='same')(input_img)
x = Conv2D(15, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Conv2D(20, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(30, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#upsampling
x = Conv2D(30, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((3, 3))(x)
x = Conv2D(25, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(20, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)

#model.add(UpSampling2D((2, 2)))
x = Conv2D(1, (3, 3), padding='same')(x)
out = Activation('sigmoid')(x)

model = Model(input_img, out)
model.compile(loss="binary_crossentropy", 
              optimizer='adam',
             metrics=['accuracy'])

model.fit(X, Y, epochs=5, batch_size=5)
loss_and_metrics = model.evaluate(Xte,Yte)

print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))
