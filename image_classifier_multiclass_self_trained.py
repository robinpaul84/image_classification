from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import callbacks
import matplotlib.pyplot as plt
import numpy as np
# dimensions of images.
img_width, img_height = 229, 229
model_name = "cartoon_noncartoon_T4700v1391_cleaned_selftrained_softamx_echo10_dropout"

train_data_dir = 'cartoon_noncartoon/train'
validation_data_dir = 'cartoon_noncartoon/validation'


epochs = 10
batch_size = 32
import glob
nb_classes = len(glob.glob(train_data_dir + "/*"))

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

csv_logger = callbacks.CSVLogger('training.log', separator=',', append=False)


def plot_training(history):

    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_name + ".png")

    plt.show()


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
#added dropout for reducing overfit
model.add(Dropout(0.5))
model.add(Dense(64)) # try 128 as the hidden layer between flatten and dense
model.add(Activation('relu'))
model.add(Dropout(0.5))
print("No of classes : ",nb_classes)
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


label_map = (train_generator.class_indices)
print(label_map)
print("train_generator")
np.save(model_name+'class_indices.npy', label_map)

nb_train_samples = len(train_generator.filenames)
print("nb_train_samples :",nb_train_samples)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
nb_validation_samples = len(validation_generator.filenames)
label_map = (validation_generator.class_indices)
print("nb_validation_samples :",nb_validation_samples)

print(label_map)
print("validation_generator")

history_ft = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[csv_logger]
    )
print("fit_generator")
model.save(model_name+".h5")
model.save_weights(model_name+".h5py")
plot_training(history_ft)