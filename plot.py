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

#上記にmodelを記入
from keras.utils import plot_model
plot_model(model, to_file='./img/model.png', show_shapes=True)
