import os,glob,sys
from PIL import Image
from math import exp,pi,sin,cos
import numpy as np

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

train_path = glob.glob("/Users/shgtkmt/workspace/cnn_fourier/train//*.tif")
data_num = (len(train_path) - 1) - 26
X = myfourier(train_path[data_num])
for i in range(data_num):
    X = np.append(X,myfourier(train_path[i]),axis=0)
    
test_path = glob.glob("/Users/shgtkmt/workspace/cnn_fourier/train//*.tif")
data_num = len(test_path)-1
Xte = myfourier(test_path[data_num])
for i in range(data_num - 150):
    Xte = np.append(Xte,myfourier(test_path[i + 150]),axis=0)
    
print(X.shape,Xte.shape)

with open("./../X.txt", "w") as f:
    for num in range(X.shape[0]):
        f.write('flag0\n')
        for row in range(X.shape[1]):
            f.write('flag1\n')
            for column in range(X.shape[2]):
                f.write('flag2\n')
                for ch in range(X.shape[3]):
                    f.write(str(X[num,row,column,ch]))
                    f.write('\n')
                    
with open("./../Xte.txt", "w") as f:
    for num in range(Xte.shape[0]):
        f.write('flag0\n')
        for row in range(Xte.shape[1]):
            f.write('flag1\n')
            for column in range(Xte.shape[2]):
                f.write('flag2\n')
                for ch in range(Xte.shape[3]):
                    f.write(str(Xte[num,row,column,ch]))
                    f.write('\n')