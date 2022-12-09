# USES TENSORFLOW 1.9
import numpy as np
import pickle
import tensorflow as tf
import gc
import tensorflow.keras.layers
from tensorflow.keras.layers import Dense, TimeDistributed
from tensorflow.keras import models, Model
import helping_functions as hf
import layers.layers_rescaling as rs

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow


def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier  # this is from global space - change this as you need
    except:
        pass

    print("resetting: ")
    print(gc.collect())  # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))


def get_full_save_path(image_size, save_path, **kwargs):
    return save_path + "RNN_" + str(image_size) + "_" + str(kwargs["seq_len"]) + "_" + str(kwargs["cell"][1])


def train_RNN(dataset_path, image_size, save_path, **kwargs):
    cnn_model_path = kwargs["cnn_model_path"]
    seq_len = kwargs["seq_len"]
    cell = kwargs["cell"][0]
    freeze = kwargs["freeze"]
    full_save_path = get_full_save_path(image_size, save_path, **kwargs)

    train_percentage = 0.8
    val_percentage = 0.1
    test_percentage = 0.1
    file_names, images, labels, file_negative_positive = hf.load_amfed_inputs_from_file(dataset_path, image_size,
                                                                                        every_nth=3)

    np.random.shuffle(file_names)
    len_files = len(file_names)
    training_files = file_names[:int(len_files * train_percentage)]
    validation_files = file_names[
                       int(len_files * train_percentage):int(len_files * (train_percentage + val_percentage))]
    test_files = file_names[int(len_files * (train_percentage + val_percentage)):int(
        len_files * (train_percentage + val_percentage + test_percentage))]

    length_of_matrix = 0
    for file in training_files:
        length_of_matrix += labels[file].shape[0] // seq_len

    train_sequences = np.empty((length_of_matrix, seq_len, image_size, image_size, 3), dtype='int8')
    train_labels = np.empty((length_of_matrix))
    idx = 0
    for file in training_files:
        temp_img = images[file].astype('int8')
        current_labels = labels[file]
        for i in range(images[file].shape[0] // seq_len):
            train_sequences[idx, :, :, :, :] = temp_img[i * seq_len: (i + 1) * seq_len]
            train_labels[idx] = current_labels[(i + 1) * seq_len - 1]
            idx += 1
    number_neg_pos = [train_labels.shape[0] - train_labels.sum(), train_labels.sum()]

    class_weight = {0: (1 / number_neg_pos[0]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0),
                    1: (1 / number_neg_pos[1]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)}
    train_labels = tf.keras.utils.to_categorical(train_labels)
    # validation---------------------------------------------
    length_of_matrix = 0
    for file in validation_files:
        length_of_matrix += labels[file].shape[0] // seq_len

    val_sequences = np.empty((length_of_matrix, seq_len, image_size, image_size, 3), dtype='int8')
    val_labels = np.empty((length_of_matrix))
    idx = 0
    for file in validation_files:
        temp_img = images[file].astype('int8')
        current_labels = labels[file]
        for i in range(images[file].shape[0] // seq_len):
            val_sequences[idx, :, :, :, :] = temp_img[i * seq_len: (i + 1) * seq_len]
            val_labels[idx] = current_labels[(i + 1) * seq_len - 1]
            idx += 1
    val_labels = tf.keras.utils.to_categorical(val_labels)
    # test----------------------------------------------
    length_of_matrix = 0
    for file in test_files:
        length_of_matrix += labels[file].shape[0] // seq_len

    test_sequences = np.empty((length_of_matrix, seq_len, image_size, image_size, 3), dtype='int8')
    test_labels = np.empty((length_of_matrix))
    idx = 0
    for file in test_files:
        temp_img = images[file].astype('int8')
        current_labels = labels[file]
        for i in range(images[file].shape[0] // seq_len):
            test_sequences[idx, :, :, :, :] = temp_img[i * seq_len: (i + 1) * seq_len]
            test_labels[idx] = current_labels[(i + 1) * seq_len - 1]
            idx += 1
    test_labels = tf.keras.utils.to_categorical(test_labels)

    cnn_model = tf.keras.models.load_model(cnn_model_path, custom_objects={'Rescaling': rs.Rescaling})

    if freeze:
        for layer in cnn_model.layers[:-5]:
            layer.trainable = False
        for layer in cnn_model.layers[-5:]:
            layer.trainable = True

    new_model = Model(inputs=cnn_model.layers[0].input, outputs=cnn_model.layers[-3].output)
    time = models.Sequential()
    time.add(TimeDistributed(new_model))
    time.add(cell(20))
    time.add(Dense(2, activation="softmax"))

    checkpoint_filepath = full_save_path + "_model.hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_acc',
        save_best_only=True)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=0,
        patience=15,
        verbose=0,
        mode="auto",
        restore_best_weights=True)

    time.compile(optimizer='adam',
                 loss=tf.keras.losses.categorical_crossentropy,
                 metrics=['accuracy'])

    history = time.fit(train_sequences,
                       train_labels,
                       epochs=100,
                       validation_data=(val_sequences, val_labels),
                       callbacks=[early_stopping_callback, model_checkpoint_callback],
                       class_weight=class_weight,
                       shuffle=True)

    file = open(full_save_path + "_history.pkl", 'wb')
    pickle.dump(history.history, file)
    pickle.dump(hf.get_confusion_matrix(test_labels, time.predict(test_sequences)), file)
    pickle.dump({"training_files": training_files, "validation_files": validation_files, "test_files": test_files},
                file)

    file.close()

    tf.keras.backend.clear_session()
    del time
    del cnn_model
    del new_model
    gc.collect()
    reset_keras()
