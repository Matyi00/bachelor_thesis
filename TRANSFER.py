##
import pickle
import tensorflow as tf
import gc
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import models, Model

import helping_functions as hf

import layers.layers_random_translation as randomTranslation
import layers.layers_random_contrast as randomContrast
import layers.layers_random_flip as randomFlip
import layers.layers_rescaling as rescaling


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(0)#todo


##
def get_full_save_path(image_size, save_path, **kwargs):
    return save_path + "TRANSFER_" + str(kwargs["dataset"]) + "_" + str(image_size) + "_" + str(
        kwargs["model"]) + "_" + str(
        kwargs["contrast"]) + "_" + str(kwargs["flip"]) + "_" + str(kwargs["translation"])

##
def train_TRANSFER(dataset_path, image_size, save_path, **kwargs):
    model = kwargs["model"]
    contrast = kwargs["contrast"]
    flip = kwargs["flip"]
    translation = kwargs["translation"]
    dataset = kwargs["dataset"]
    if dataset != "genki" and dataset != "amfed":
        raise Exception("Wrong dataset given: must be either genki or amfed")
    if model != "resnet" and model != "xception":
        raise Exception("Wrong model given: must be either resnet or xception")

    if dataset == "genki":
        x_train, y_train, x_val, y_val, x_test, y_test, class_weight, order = hf.load_genki_from_file_with_shuffle(
            dataset_path, image_size, 0.8, 0.1, 0.1)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, class_weight, order = hf.load_amfed_from_file_with_shuffle(
            dataset_path, image_size, 0.8, 0.1, 0.1, every_nth=3)

    if model == "resnet":
        transfer_model = tf.keras.applications.ResNet50V2
    else:
        transfer_model = tf.keras.applications.Xception

    full_save_path = get_full_save_path(image_size, save_path, **kwargs)

    base_model = transfer_model(weights='imagenet', include_top=False,
                                                  input_shape=(image_size, image_size, 3), pooling='max')

    for layer in base_model.layers:
        layer.trainable = False
    # -----------------------------------------------
    input_layer = tf.keras.Input(shape=(image_size, image_size, 3), name="myInput")
    preprocessing_layer_list = list()
    preprocessing_layer_list.append(rescaling.Rescaling(1 / 127.5, offset=-1))
    if flip:
        preprocessing_layer_list.append(randomFlip.RandomFlip(mode="horizontal"))
    if contrast != 0:
        preprocessing_layer_list.append(randomContrast.RandomContrast(contrast))
    if translation != 0:
        preprocessing_layer_list.append(randomTranslation.RandomTranslation(translation, translation))
    img_augmentation = tf.keras.Sequential(preprocessing_layer_list, name="img_augmentation")
    augmented = img_augmentation(input_layer)

    y = base_model(augmented)
    x = Flatten()(y)

    # ------------------------------------------------

    x = Dropout(0.2)(x)
    predictions = Dense(2, activation="softmax")(x)

    checkpoint_filepath = full_save_path + "_nonfine_model.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=15,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True)

    head_model = Model(inputs=input_layer, outputs=predictions)
    head_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    history_1 = head_model.fit(x_train,
                               y_train,
                               epochs=80,
                               validation_data=(x_val, y_val),
                               callbacks=[early_stopping_callback,model_checkpoint_callback],
                               class_weight=class_weight)

    file = open(full_save_path + "_nonfine_history.pkl", 'wb')
    pickle.dump(history_1.history, file)
    pickle.dump(hf.get_confusion_matrix(y_test, head_model.predict(x_test)), file)
    pickle.dump(order, file)
    file.close()

    checkpoint_filepath = full_save_path + "_fine_model.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        save_best_only=True)

    head_model.layers[2].trainable = True
    for layer in head_model.layers[2].layers:
        layer.trainable = True
    head_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])
    history_2 = head_model.fit(x_train,
                               y_train,
                               epochs=80,
                               validation_data=(x_test, y_test),
                               callbacks=[early_stopping_callback, model_checkpoint_callback],
                               class_weight=class_weight)

    file = open(full_save_path + "_fine_history.pkl", 'wb')
    pickle.dump(history_2.history, file)
    pickle.dump(hf.get_confusion_matrix(y_test, head_model.predict(x_test)), file)
    pickle.dump(order, file)
    file.close()

    tf.keras.backend.clear_session()
    del head_model
    gc.collect()
