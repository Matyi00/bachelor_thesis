##
import pickle
import tensorflow as tf
import gc
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten
from tensorflow.keras import models


import helping_functions as hf

import layers.layers_random_translation as randomTranslation
import layers.layers_random_contrast as randomContrast
import layers.layers_random_flip as randomFlip
import layers.layers_rescaling as rescaling


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(0)


##
def get_full_save_path(image_size, save_path, **kwargs):
    return save_path + "FNN_" + str(kwargs["dataset"]) + "_" + str(image_size) + "_" + str(
        kwargs["layer_nb"]) + "_" + str(
        kwargs["layer_size"]) + "_" + str(kwargs["dropout"]) + "_" + str(kwargs["activation"][1]) + "_" + str(
        kwargs["contrast"]) + "_" + str(kwargs["flip"]) + "_" + str(kwargs["translation_x"]) + "_" + str(
        kwargs["translation_y"])


##
def train_FNN(dataset_path, image_size, save_path, **kwargs):
    layer_nb = kwargs["layer_nb"]
    layer_size = kwargs["layer_size"]
    dropout = kwargs["dropout"]
    activation = kwargs["activation"][0]
    contrast = kwargs["contrast"]
    flip = kwargs["flip"]
    translation_x = kwargs["translation_x"]
    translation_y = kwargs["translation_y"]
    dataset = kwargs["dataset"]
    if dataset != "genki" and dataset != "amfed":
        raise Exception("Wrong dataset given: must be either genki or amfed")

    if dataset == "genki":
        x_train, y_train, x_val, y_val, x_test, y_test, class_weight, order = hf.load_genki_from_file_with_shuffle(
            dataset_path, image_size, 0.8, 0.1, 0.1)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, class_weight, order = hf.load_amfed_from_file_with_shuffle(
            dataset_path, image_size, 0.8, 0.1, 0.1, every_nth=3)

    full_save_path = get_full_save_path(image_size, save_path, **kwargs)



    preprocessing_layer_list = list()
    preprocessing_layer_list.append(rescaling.Rescaling(1/255., offset=0))
    if flip:
        preprocessing_layer_list.append(randomFlip.RandomFlip(mode="horizontal"))
    if contrast != 0:
        preprocessing_layer_list.append(randomContrast.RandomContrast(contrast))
    if translation_x != 0 or translation_y != 0:
        preprocessing_layer_list.append(randomTranslation.RandomTranslation(translation_x, translation_y))
    img_augmentation = tf.keras.Sequential(preprocessing_layer_list, name="img_augmentation")

    model = models.Sequential()
    if (len(preprocessing_layer_list)) > 0:
        model.add(img_augmentation)
    model.add(InputLayer(input_shape=(image_size, image_size, 3)))

    model.add(Flatten())
    for i in range(layer_nb):
        model.add(Dense(layer_size, activation=activation))
        if dropout != 0:
            model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))

    checkpoint_filepath = full_save_path + "_model.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0,
        patience=20,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(x_train,
                        y_train,
                        epochs=100,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping_callback, model_checkpoint_callback],
                        class_weight=class_weight)

    file = open(full_save_path + "_history.pkl", 'wb')
    pickle.dump(history.history, file)
    pickle.dump(hf.get_confusion_matrix(y_test, model.predict(x_test)), file)
    pickle.dump(order, file)
    file.close()

    tf.keras.backend.clear_session()
    del model
    gc.collect()
