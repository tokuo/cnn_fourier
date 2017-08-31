from keras.models import Sequential
model = Sequential()
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
model.add(Conv2D(20, (7, 7), padding='same', input_shape=(240, 240, 5)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(40, (5, 5), padding='same',))#strides
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
#upsampling
model.add(Conv2D(30, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D((4, 4)))
model.add(Conv2D(10, (7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D((3, 3)))

model.add(Flatten())
#model.add(Dense(240*240*5))
#model.add(LeakyReLU(0.2))
#model.add(Dropout(0.5))
model.add(Dense(240*240))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", 
              optimizer='adam',
             metrics=['accuracy'])

#上記にmodelを記入
from keras.utils import plot_model
plot_model(model, to_file='./demo/model.png', show_shapes=True)
