import tensorflow as tf
import tensorflow.keras as keras
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate
from keras.optimizer_v2 import adam

def unet(input_size=(256, 256, 3)):

    inputs = Input(input_size)

    # 1st down block
    conv1 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)

    conv1 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 2nd block
    conv2 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool1)

    conv2 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 3rd block
    conv3 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool2)

    conv3 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 4th block
    conv4 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool3)

    conv4 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # 5th block
    conv5 = Conv2D(1024,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(pool4)

    conv5 = Conv2D(1024,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)

    # 1st up block
    up6 = Conv2D(512,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))

    merge6 = Concatenate([drop4, up6], axis=3)

    conv6 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv6)

    # 2nd up block
    up7 = Conv2D(256,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))

    merge7 = Concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge7)

    conv7 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv7)

    # 3rd up block
    up8 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))

    merge8 = Concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge8)

    conv8 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv8)

    # 4th up block
    up9 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))

    merge9 = Concatenate([conv1, up9], axis=3)

    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(merge9)

    conv9 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(2,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv9)

    # 5th up block
    conv10 = Conv2D(4, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #model.summary()
    
    return model
