from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
from tensorflow import concat
from tensorflow.keras.optimizers import Adam


def vgg16_unet(input_shape=(128, 128, 3)):

    # Block 1
    img_input = Input(shape=input_shape)

    block1_conv1 = Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block1_conv1")(img_input)
    block1_conv2 = Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block1_conv2")(block1_conv1)
    block1_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block1_pool")(block1_conv2)

    # Block 2
    block2_conv1 = Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block2_conv1")(block1_pool)
    block2_conv2 = Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block2_conv2")(block2_conv1)
    block2_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block2_pool")(block2_conv2)

    # Block 3
    block3_conv1 = Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block3_conv1")(block2_pool)
    block3_conv2 = Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block3_conv2")(block3_conv1)
    block3_conv3 = Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block3_conv3")(block3_conv2)
    block3_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block3_pool")(block3_conv3)

    # Block 4
    block4_conv1 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block4_conv1")(block3_pool)
    block4_conv2 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block4_conv2")(block4_conv1)
    block4_conv3 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block4_conv3")(block4_conv2)
    block4_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name="block4_pool")(block4_conv3)

    # Block 5
    block5_conv1 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block5_conv1")(block4_pool)
    block5_conv2 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block5_conv2")(block5_conv1)
    block5_conv3 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block5_conv3")(block5_conv2)
    block5_pool = MaxPooling2D((2, 2), strides=(2, 2),
                               name='block5_pool')(block5_conv3)

    # Espace latent
    #block5_global_pool = GlobalAveragePooling2D(keepdims=True)(block5_pool)

    # Block 6
    block6_conv1 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block6_conv1")(
                              UpSampling2D(size=(2, 2))(block5_pool))
    #block6_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name="block6_conv2")(UpSampling2D(size = (2,2))(block5_global_pool))
    #block6_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name="block6_conv1")(UpSampling2D(size = (2,2))(block5_global_pool))
    #block6_conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name="block6_conv1")(UpSampling2D(size = (2,2))(block5_global_pool))
    block6_merge = concat([block5_conv3, block6_conv1], 3)
    block6_conv2 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block6_conv2")(block6_merge)
    block6_conv3 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block6_conv3")(block6_conv2)

    # Block 7
    block7_conv1 = Conv2D(512, (2, 2),
                          activation='relu',
                          padding='same',
                          name="block7_conv1")(
                              UpSampling2D(size=(2, 2))(block6_conv3))
    block7_merge = concat([block4_conv3, block7_conv1], 3)
    block7_conv2 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block7_conv2")(block7_merge)
    block7_conv3 = Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block7_conv3")(block7_conv2)

    # Block 8
    block8_conv1 = Conv2D(256, (2, 2),
                          activation='relu',
                          padding='same',
                          name="block8_conv1")(
                              UpSampling2D(size=(2, 2))(block7_conv3))
    block8_merge = concat([block3_conv3, block8_conv1], 3)
    block8_conv2 = Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block8_conv2")(block8_merge)
    block8_conv3 = Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block8_conv3")(block8_conv2)

    # Block 9
    block9_conv1 = Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block9_conv1")(
                              UpSampling2D(size=(2, 2))(block8_conv3))
    block9_merge = concat([block2_conv2, block9_conv1], 3)
    block9_conv2 = Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name="block9_conv2")(block9_merge)

    # Block 10
    block10_conv1 = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           name="block10_conv1")(
                               UpSampling2D(size=(2, 2))(block9_conv2))
    block10_merge = concat([block1_conv2, block10_conv1], 3)
    block10_conv2 = Conv2D(64, (3, 3),
                           activation='relu',
                           padding='same',
                           name="block10_conv2")(block10_merge)

    # Block 11
    output = Conv2D(4, 1, activation="sigmoid")(block10_conv2)

    model = Model(inputs=img_input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-04),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model
