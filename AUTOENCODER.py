##
import tensorflow as tf
import pickle

from tensorflow.keras import Input

import layers.layers_rescaling as rs
import helping_functions as hf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 224


def get_full_save_path(save_path, **kwargs):
    return save_path + "AUTOENCODER_" + str(kwargs["latent_dimension"])


def train_AUTOENCODER(dataset_path, save_path, **kwargs):
    training_images, validation_images, test_images = hf.load_autoencoder_dataset_from_file(dataset_path, IMG_SIZE, 0.8, 0.1, 0.1)

    latent_dimension = kwargs["latent_dimension"]

    full_save_path = get_full_save_path(save_path, **kwargs)

    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = rs.Rescaling(1 / 255., offset=0)(input_img)
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same', name="encoded")(x)
    x = Flatten()(x)
    encoded = Dense(latent_dimension)(x)
    x = Dense(14 * 14 * 32)(encoded)
    x = Reshape(target_shape=(14, 14, 32))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='MSE')
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True
    )
    checkpoint_filepath = full_save_path + "_model.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        save_best_only=True)
    autoencoder.summary()

    history = autoencoder.fit(training_images,
                              training_images / 255.,
                              epochs=60,
                              validation_data=(validation_images, validation_images / 255.),
                              callbacks=[early_stopping_callback, model_checkpoint_callback])

    file = open(full_save_path + "_history.pkl", 'wb')
    pickle.dump(history.history, file)
    file.close()
