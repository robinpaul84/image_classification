from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import utils
from keras import callbacks
csv_logger = callbacks.CSVLogger('training.log', separator=',', append=False)

script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, 'cartoon_noncartoon/train')
test_set_path = os.path.join(script_dir, 'cartoon_noncartoon/train')

# Image dimensions
img_width, img_height = 150, 150

"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""


def create_model(p, input_shape=(img_width, img_height, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p / 2))
    model.add(Dense(2, activation='softmax'))

    # Compiling the CNN
    optimizer =  'rmsprop' #Adam(lr=1e-3)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
    return model


"""
    Fitting the CNN to the images.
"""


def run_training(bs=32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(training_set_path,
                                                     target_size=(img_width, img_height),
                                                     batch_size=bs,
                                                     class_mode='categorical')
    nb_train_samples = len(training_set.filenames)

    test_set = test_datagen.flow_from_directory(test_set_path,
                                                target_size=(img_width, img_height),
                                                batch_size=bs,
                                                class_mode='categorical')

    nb_validation_samples = len(test_set.filenames)
    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))
    history_ft = model.fit_generator(training_set,
                        steps_per_epoch=nb_train_samples / bs,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=nb_validation_samples / bs,
                        callbacks=[csv_logger]
                        )
    model_name = 'udemy_self_trained'
    model.save( model_name+ ".h5")
    model.save_weights(model_name + ".h5py")
    utils.plot_training(history=history_ft)


def main():
    run_training(bs=32, epochs=10)


""" Main """
if __name__ == "__main__":
    main()