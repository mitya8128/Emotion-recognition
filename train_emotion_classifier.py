"""
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import mini_XCEPTION, big_XCEPTION, big_multi_XCEPTION, VGG_16_modified, multi_VGG_16_modified
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cores", help="define number of CPU cores",
                    type=int)
parser.add_argument("--epoches", help="number of training epoches",
                    type=int)
args = parser.parse_args()
if args.cores:
    print('number of cores is {}' .format("--cores"))

# parameters
batch_size = 32
num_epochs = args.epoches
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'
model_name = 'big_multi_XCEPTION'
number_of_cores = args.cores
gpu_use = False    # switch between gpu and cpu use


if gpu_use:

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

else:

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = tf.compat.v1.ConfigProto(#intra_op_parallelism_threads=number_of_cores,
                            #inter_op_parallelism_threads=number_of_cores,
                            #allow_soft_placement=True,
                            device_count={'CPU': number_of_cores})
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = big_multi_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
#plot_model(model, to_file='model.png')    # need Graphviz



class TimeHistory(Callback):

    def __init__(self):
        super().__init__()
        self.times = []

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)

 # callbacks

log_file_path = base_path + '_emotion_training_{}_{}_{}.log' .format(num_epochs,number_of_cores,model_name)
csv_logger = CSVLogger(log_file_path, append=False)
#early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
trained_models_path = base_path + model_name
model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)

time_callback = TimeHistory()
callbacks = [model_checkpoint, csv_logger, reduce_lr, time_callback]

# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True)
history = model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))

# training time
times = time_callback.times   # epoch computation times
ellapsed_time = sum(times)    # overall ellapsed time
print('training time for each epoch: {}' .format(times))
print('overall ellapsed time: {}' .format(ellapsed_time))

# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy_number of cores:{},_ellapsed_time:{},_model_name:{}' .format(number_of_cores,ellapsed_time,
                                                                                       model_name))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_accuracy_{}ep_{}.png' .format(num_epochs,model_name))
#plt.show()
#plt.close()

# Plot training & validation loss values
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss_number of cores:{},_ellapsed_time:{},_model_name:{}' .format(number_of_cores,ellapsed_time,
                                                                                   model_name))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('model_loss_{}ep_{}.png' .format(num_epochs,model_name))
#plt.show()
#plt.close()