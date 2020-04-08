from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D, Cropping2D, Average
from keras import layers
from keras.regularizers import l2

def simple_CNN(input_shape, num_classes):

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax',name='predictions'))
    return model

def simpler_CNN(input_shape, num_classes):

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(5, 5),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))

    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax',name='predictions'))
    return model

def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(8, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(8, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(8, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model


def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=True)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=True)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=True)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=True)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model

def big_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=True)(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=True)(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=True)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=True)(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=True)(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=True)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=True)(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=True)(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model

# big_XCEPTION with multilocal feature learning
def big_multi_XCEPTION(input_shape, num_classes):
    img_input = Input(input_shape)
    eyes = Cropping2D(((0,24),(0,0)))(img_input)
    mouth = Cropping2D(((24,0),(0,0)))(img_input)
    sub_models = []

    # whole face
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)    # x for whole image, y for eyes, z for mouth
    x = BatchNormalization(name='block1_conv1_bn_face')(x)
    x = Activation('relu', name='block1_conv1_act_face')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization(name='block1_conv2_bn_face')(x)
    x = Activation('relu', name='block1_conv2_act_face')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv1_bn_face')(x)
    x = Activation('relu', name='block2_sepconv2_act_face')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block2_sepconv2_bn_face')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act_face')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv1_bn_face')(x)
    x = Activation('relu', name='block3_sepconv2_act_face')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization(name='block3_sepconv2_bn_face')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])
    x = Conv2D(num_classes, (3, 3),
            #kernel_regularizer=regularization,
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output_face = Activation('softmax',name='predictions_face')(x)


    # eyes
    y = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(eyes)  # x for whole image, y for eyes, z for mouth
    y = BatchNormalization(name='block1_conv1_bn_eyes')(y)
    y = Activation('relu', name='block1_conv1_act_eyes')(y)
    y = Conv2D(64, (3, 3), use_bias=False)(y)
    y = BatchNormalization(name='block1_conv2_bn_eyes')(y)
    y = Activation('relu', name='block1_conv2_act_eyes')(y)

    residual_eyes = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(y)
    residual_eyes = BatchNormalization()(residual_eyes)

    y = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(y)
    y = BatchNormalization(name='block2_sepconv1_bn_eyes')(y)
    y = Activation('relu', name='block2_sepconv2_act_eyes')(y)
    y = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(y)
    y = BatchNormalization(name='block2_sepconv2_bn_eyes')(y)

    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)
    y = layers.add([y, residual_eyes])

    residual_eyes = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(y)
    residual_eyes = BatchNormalization()(residual_eyes)

    y = Activation('relu', name='block3_sepconv1_act_eyes')(y)
    y = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(y)
    y = BatchNormalization(name='block3_sepconv1_bn_eyes')(y)
    y = Activation('relu', name='block3_sepconv2_act_eyes')(y)
    y = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(y)
    y = BatchNormalization(name='block3_sepconv2_bn_eyes')(y)

    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)
    y = layers.add([y, residual_eyes])
    y = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(y)
    y = GlobalAveragePooling2D()(y)
    output_eyes = Activation('softmax', name='predictions_eyes')(y)

    # mouth
    z = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(mouth)  # x for whole image, y for eyes, z for mouth
    z = BatchNormalization(name='block1_conv1_bn_mouth')(z)
    z = Activation('relu', name='block1_conv1_act_mouth')(z)
    z = Conv2D(64, (3, 3), use_bias=False)(z)
    z = BatchNormalization(name='block1_conv2_bn_mouth')(z)
    z = Activation('relu', name='block1_conv2_act_mouth')(z)

    residual_mouth = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(z)
    residual_mouth = BatchNormalization()(residual_mouth)

    z = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(z)
    z = BatchNormalization(name='block2_sepconv1_bn_mouth')(z)
    z = Activation('relu', name='block2_sepconv2_act_mouth')(z)
    z = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(z)
    z = BatchNormalization(name='block2_sepconv2_bn_mouth')(z)

    z = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(z)
    z = layers.add([z, residual_mouth])

    residual_mouth = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(z)
    residual_mouth = BatchNormalization()(residual_mouth)

    z = Activation('relu', name='block3_sepconv1_act_mouth')(z)
    z = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(z)
    z = BatchNormalization(name='block3_sepconv1_bn_mouth')(z)
    z = Activation('relu', name='block3_sepconv2_act_mouth')(z)
    z = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(z)
    z = BatchNormalization(name='block3_sepconv2_bn_mouth')(z)

    z = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(z)
    z = layers.add([z, residual_mouth])
    z = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(z)
    z = GlobalAveragePooling2D()(z)
    output_mouth = Activation('softmax', name='predictions_mouth')(z)

    # model averaging
    sub_models = [output_face,output_eyes,output_mouth]
    output = Average()
    output = output(sub_models)

    model = Model(img_input, output)
    return model

def VGG_16_modified(input_shape, num_classes):
    img_input = Input(input_shape)
    eyes = Cropping2D(((0,24),(0,0)))(img_input)
    mouth = Cropping2D(((24,0),(0,0)))(img_input)
    sub_models = []

    # block1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False,padding='same')(img_input)    # x for whole image, y for eyes, z for mouth
    x = BatchNormalization(name='block1_conv1_bn_face')(x)
    x = Activation('relu', name='block1_conv1_act_face')(x)
    x = Conv2D(32, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block1_conv2_bn_face')(x)
    x = Activation('relu', name='block1_conv2_act_face')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block2
    x = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block2_conv1_bn_face')(x)
    x = Activation('relu', name='block2_conv1_act_face')(x)
    x = Conv2D(64, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block2_conv2_bn_face')(x)
    x = Activation('relu', name='block2_conv2_act_face')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block3
    x = Conv2D(128, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block3_conv1_bn_face')(x)
    x = Activation('relu', name='block3_conv1_act_face')(x)
    x = Conv2D(128, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block3_conv2_bn_face')(x)
    x = Activation('relu', name='block3_conv2_act_face')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # block4
    x = Conv2D(256, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block4_conv1_bn_face')(x)
    x = Activation('relu', name='block4_conv1_act_face')(x)
    x = Conv2D(256, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block4_conv2_bn_face')(x)
    x = Activation('relu', name='block4_conv2_act_face')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # block5
    x = Conv2D(512, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block5_conv1_bn_face')(x)
    x = Activation('relu', name='block5_conv1_act_face')(x)
    x = Conv2D(512, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block5_conv2_bn_face')(x)
    x = Activation('relu', name='block5_conv2_act_face')(x)

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output_face = Activation('softmax', name='predictions_face')(x)

    model = Model(img_input,output_face)

    return model

def multi_VGG_16_modified(input_shape, num_classes):
    img_input = Input(input_shape)
    eyes = Cropping2D(((0,24),(0,0)))(img_input)
    mouth = Cropping2D(((24,0),(0,0)))(img_input)
    sub_models = []

    # whole face
    # block1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False,padding='same')(img_input)    # x for whole image, y for eyes, z for mouth
    x = BatchNormalization(name='block1_conv1_bn_face')(x)
    x = Activation('relu', name='block1_conv1_act_face')(x)
    x = Conv2D(32, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block1_conv2_bn_face')(x)
    x = Activation('relu', name='block1_conv2_act_face')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block2
    x = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block2_conv1_bn_face')(x)
    x = Activation('relu', name='block2_conv1_act_face')(x)
    x = Conv2D(64, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block2_conv2_bn_face')(x)
    x = Activation('relu', name='block2_conv2_act_face')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block3
    x = Conv2D(128, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block3_conv1_bn_face')(x)
    x = Activation('relu', name='block3_conv1_act_face')(x)
    x = Conv2D(128, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block3_conv2_bn_face')(x)
    x = Activation('relu', name='block3_conv2_act_face')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # block4
    x = Conv2D(256, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block4_conv1_bn_face')(x)
    x = Activation('relu', name='block4_conv1_act_face')(x)
    x = Conv2D(256, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block4_conv2_bn_face')(x)
    x = Activation('relu', name='block4_conv2_act_face')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # block5
    x = Conv2D(512, (3, 3), strides=(2, 2), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block5_conv1_bn_face')(x)
    x = Activation('relu', name='block5_conv1_act_face')(x)
    x = Conv2D(512, (3, 3), use_bias=False,padding='same')(x)
    x = BatchNormalization(name='block5_conv2_bn_face')(x)
    x = Activation('relu', name='block5_conv2_act_face')(x)

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output_face = Activation('softmax', name='predictions_face')(x)

    # eyes
    # block1
    y = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same')(eyes)  # x for whole image, y for eyes, z for mouth
    y = BatchNormalization(name='block1_conv1_bn_eyes')(y)
    y = Activation('relu', name='block1_conv1_act_eyes')(y)
    y = Conv2D(32, (3, 3), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block1_conv2_bn_eyes')(y)
    y = Activation('relu', name='block1_conv2_act_eyes')(y)

    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)

    # block2
    y = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block2_conv1_bn_eyes')(y)
    y = Activation('relu', name='block2_conv1_act_eyes')(y)
    y = Conv2D(64, (3, 3), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block2_conv2_bn_eyes')(y)
    y = Activation('relu', name='block2_conv2_act_eyes')(y)

    y = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(y)

    # block3
    y = Conv2D(128, (3, 3), strides=(2, 2), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block3_conv1_bn_eyes')(y)
    y = Activation('relu', name='block3_conv1_act_eyes')(y)
    y = Conv2D(128, (3, 3), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block3_conv2_bn_eyes')(y)
    y = Activation('relu', name='block3_conv2_act_eyes')(y)

    y = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(y)

    # block4
    y = Conv2D(256, (3, 3), strides=(2, 2), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block4_conv1_bn_eyes')(y)
    y = Activation('relu', name='block4_conv1_act_eyes')(y)
    y = Conv2D(256, (3, 3), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block4_conv2_bn_eyes')(y)
    y = Activation('relu', name='block4_conv2_act_eyes')(y)

    y = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(y)

    # block5
    y = Conv2D(512, (3, 3), strides=(2, 2), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block5_conv1_bn_eyes')(y)
    y = Activation('relu', name='block5_conv1_act_eyes')(y)
    y = Conv2D(512, (3, 3), use_bias=False, padding='same')(y)
    y = BatchNormalization(name='block5_conv2_bn_eyes')(y)
    y = Activation('relu', name='block5_conv2_act_eyes')(y)

    y = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(y)
    y = GlobalAveragePooling2D()(y)
    output_eyes = Activation('softmax', name='predictions_eyes')(y)

    # mouth
    # block1
    z = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, padding='same')(mouth)
    z = BatchNormalization(name='block1_conv1_bn_mouth')(z)
    z = Activation('relu', name='block1_conv1_act_mouth')(z)
    z = Conv2D(32, (3, 3), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block1_conv2_bn_mouth')(z)
    z = Activation('relu', name='block1_conv2_act_mouth')(z)

    z = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(z)

    # block2
    z = Conv2D(64, (3, 3), strides=(2, 2), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block2_conv1_bn_mouth')(z)
    z = Activation('relu', name='block2_conv1_act_mouth')(z)
    z = Conv2D(64, (3, 3), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block2_conv2_bn_mouth')(z)
    z = Activation('relu', name='block2_conv2_act_mouth')(z)

    z = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(z)

    # block3
    z = Conv2D(128, (3, 3), strides=(2, 2), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block3_conv1_bn_mouth')(z)
    z = Activation('relu', name='block3_conv1_act_mouth')(z)
    z = Conv2D(128, (3, 3), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block3_conv2_bn_mouth')(z)
    z = Activation('relu', name='block3_conv2_act_mouth')(z)

    z = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(z)

    # block4
    z = Conv2D(256, (3, 3), strides=(2, 2), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block4_conv1_bn_mouth')(z)
    z = Activation('relu', name='block4_conv1_act_mouth')(z)
    z = Conv2D(256, (3, 3), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block4_conv2_bn_mouth')(z)
    z = Activation('relu', name='block4_conv2_act_mouth')(z)

    z = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(z)

    # block5
    z = Conv2D(512, (3, 3), strides=(2, 2), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block5_conv1_bn_mouth')(z)
    z = Activation('relu', name='block5_conv1_act_mouth')(z)
    z = Conv2D(512, (3, 3), use_bias=False, padding='same')(z)
    z = BatchNormalization(name='block5_conv2_bn_mouth')(z)
    z = Activation('relu', name='block5_conv2_act_mouth')(z)

    z = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(z)
    z = GlobalAveragePooling2D()(z)
    output_mouth = Activation('softmax', name='predictions_mouth')(z)

    # model averaging
    sub_models = [output_face, output_eyes, output_mouth]
    output = Average()
    output = output(sub_models)
    model = Model(img_input,output)

    return model

if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    #model = tiny_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = mini_XCEPTION(input_shape, num_classes)
    #model.summary()
    #model = big_XCEPTION(input_shape, num_classes)
    #model.summary()
    model = simple_CNN((48, 48, 1), num_classes)
    model.summary()
