import glob
from PIL import Image
import numpy as np
from math import exp,pi,sin,cos

#訓練データ用
def myfourier(tiff_path):
    pre_img =  Image.open(tiff_path)
    pre_img = pre_img.convert('L')
    _x = pre_img.size[0]
    _y = pre_img.size[1]
    pre_img =  np.array(pre_img)
    X = np.zeros([1, _y, _x, 5],'float32')
    X[0,:,:,0] = pre_img
    img = np.zeros([_y+16, _x+16])
    img[8:-8, 8:-8] = pre_img
    local_f = np.zeros([16,16])
    cir_ave = 0
    for y in range(_y):
        for x in range(_x):
            local_f = img[y:y+16, x:x+16]
            #local_f = a0_rm(local_f,x,y)
            local_f = np.fft.fft2(local_f)
            local_f = np.fft.fftshift(local_f)
            for i in range(16):
                for j in range(16):
                    local_f[i,j] = np.absolute(local_f[i,j])
            X[0,y,x,4] = local_f.sum()#256
            X[0,y,x,3] = local_f[4:12, 4:12].sum()#64
            X[0,y,x,4] = (X[0,y,x,4] - X[0,y,x,3])
            X[0,y,x,2] = local_f[6:10, 6:10].sum()#16
            X[0,y,x,3] = (X[0,y,x,3] - X[0,y,x,2])
            X[0,y,x,1] = local_f[7:9, 7:9].sum()#4
            X[0,y,x,2] = (X[0,y,x,2] - X[0,y,x,1])
            X[0,y,x,1] = X[0,y,x,1]/4
            X[0,y,x,2] = X[0,y,x,2]/12
            X[0,y,x,3] = X[0,y,x,3]/48
            X[0,y,x,4] = X[0,y,x,4]/192
    return X

#教師データ用
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



# 学習用データ：検証用データ = 150：26

#学習用(訓練データ)
train_path = glob.glob("path")
data_num = (len(train_path) - 1) - 26
X = myfourier(train_path[data_num])
for i in range(data_num):
    X = np.append(X,myfourier(train_path[i]),axis=0)
#検証用(訓練データ)
test_path = glob.glob("path")
data_num = len(test_path)-1
Xte = myfourier(test_path[data_num])
for i in range(data_num - 150):
    Xte = np.append(Xte,myfourier(test_path[i + 150]),axis=0)
#学習用(教師データ)
train_ans_path = glob.glob("path")
data_num = (len(train_ans_path) - 1) - 26
Y = mask_check(train_ans_path[data_num])
for i in range(data_num):
    Y = np.append(Y,mask_check(train_ans_path[i]),axis=0)
#検証用(教師データ)
test_ans_path = glob.glob("path")
data_num = len(test_ans_path)-1
Yte = mask_check(test_ans_path[data_num])
for i in range(data_num - 150):
    Yte = np.append(Yte,mask_check(test_ans_path[i + 150]),axis=0)
    
    
    
from keras.layers import Input, Dropout, Activation
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ThresholdedReLU, LeakyReLU
from keras.layers.merge import Add

input_img = Input(shape=(240,240,5))
x = Conv2D(10, (3, 3), padding='same')(input_img)
x = Conv2D(15, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x80 = MaxPooling2D(pool_size=(3, 3))(x) #shape=(80,80,15)
x = Conv2D(20, (3, 3), padding='same')(x80)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x40 = MaxPooling2D(pool_size=(2, 2))(x) #shape=(40,40,20)
x = Conv2D(30, (3, 3), padding='same')(x40)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#upsampling
x = Conv2D(20, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x40_2 = UpSampling2D((2, 2))(x) #shape=(40,40,20)
x = Add()([x40, x40_2])
x = Conv2D(15, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)
x80_2 = UpSampling2D((2, 2))(x) #shape=(80,80,15)
x = Add()([x80, x80_2])
x = Conv2D(10, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling2D((3, 3))(x)
#fin
x = Conv2D(1, (3, 3), padding='same')(x)
x = Reshape((240,240))(x)
x = Activation('sigmoid')(x)
out = ThresholdedReLU(theta=0.5)(x)

from keras.models import Model
model = Model(input_img, out)
model.compile(loss="binary_crossentropy", 
              optimizer='adam',
             metrics=['accuracy'])

model.fit(X, Y, epochs=15, batch_size=10)



# 以下結果の表示

loss_and_metrics = model.evaluate(Xte,Yte)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))
print(model.summary())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,255))
plt.imshow(Xte[5,:,:,0], cmap='gray')
plt.show()
ans = model.predict_on_batch(Xte)
img = np.reshape(ans[5,:],(240,240))
img = scaler.fit_transform(img)
plt.imshow(img, cmap='gray')
plt.show()
plt.imshow(Yte[5,:,:], cmap='gray')
plt.show()