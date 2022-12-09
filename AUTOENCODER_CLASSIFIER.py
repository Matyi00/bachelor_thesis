##
import tensorflow as tf
import pickle
import gc
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
import helping_functions as hf
import layers.layers_rescaling as rs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 224


def get_full_save_path(save_path, **kwargs):
    return save_path + "AUTOENCODER_CLASSIFIER_" + str(kwargs["dataset"]) + "_" + str(kwargs["latent_dimension"])


def train_AUTOENCODER_CLASSIFIER(dataset_path, save_path, **kwargs):
    model_path = kwargs["model_path"]
    dataset = kwargs["dataset"]

    if dataset != "genki" and dataset != "amfed":
        raise Exception("Wrong dataset given: must be either genki or amfed")

    if dataset == "genki":
        x_train, y_train, x_val, y_val, x_test, y_test, class_weight, order = hf.load_genki_from_file_with_shuffle(
            dataset_path, IMG_SIZE, 0.8, 0.1, 0.1)
    else:
        x_train, y_train, x_val, y_val, x_test, y_test, class_weight, order = hf.load_amfed_from_file_with_shuffle(
            dataset_path, IMG_SIZE, 0.8, 0.1, 0.1, every_nth=3)

    full_save_path = get_full_save_path(save_path, **kwargs)

    model = tf.keras.models.load_model(model_path, custom_objects={'Rescaling': rs.Rescaling})

    x = Dense(2, activation="relu", name = "a")(model.layers[11].output)
    x = Dense(2, activation="relu", name = "b")(x)
    x = Dense(2, activation="softmax", name = "c")(x)
    new_model = Model(inputs=model.inputs, outputs=x)
    for layer in new_model.layers[:12]:
        layer.trainable = False
    new_model.summary()
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

    new_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
    #new_model.summary()
    history = new_model.fit(x_train,
                            y_train,
                            epochs=100,
                            validation_data=(x_val, y_val),
                            callbacks=[model_checkpoint_callback, early_stopping_callback],
                            class_weight=class_weight)
    file = open(full_save_path + "_history.pkl", 'wb')
    pickle.dump(history.history, file)
    pickle.dump(hf.get_confusion_matrix(y_test, new_model.predict(x_test)), file)
    pickle.dump(order, file)
    file.close()

    tf.keras.backend.clear_session()
    del model
    gc.collect()
