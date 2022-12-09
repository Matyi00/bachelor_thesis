##
import pickle
import os
import numpy as np


def load_amfed_inputs_from_file(save_path, dimension, every_nth=1):
    file = open(save_path + "negative_positive_ratio.pkl", 'rb')
    file_negative_positive = pickle.load(file)
    file.close()

    file = open(save_path + 'amfed_data' + str(dimension) + '.pkl', 'rb')
    file_inputs = pickle.load(file)
    file.close()

    images = dict()
    labels = dict()

    file_names = file_inputs.keys()
    for file in file_names:
        images[file] = np.array(file_inputs[file][0])[::every_nth]
        labels[file] = np.array(file_inputs[file][1])[::every_nth]
        file_inputs.pop(file)

    temp_file_names = []
    for file in file_names:
        if labels[file].shape[0] != 0:
            temp_file_names.append(file)
        else:
            images.pop(file)
            labels.pop(file)
    file_names = temp_file_names

    return file_names, images, labels, file_negative_positive


def load_amfed_from_file_with_shuffle(save_path, dimension, training_rate, validation_rate, test_rate, every_nth=1):
    file = open(save_path + "negative_positive_ratio.pkl", 'rb')
    file_negative_positive = pickle.load(file)
    file.close()

    file = open(save_path + 'amfed_data' + str(dimension) + '.pkl', 'rb')
    file_inputs = pickle.load(file)
    file.close()

    files = file_inputs.keys()
    np.random.shuffle(files)
    len_files = len(files)

    training_files = files[:int(len_files * training_rate)]
    validation_files = files[int(len_files * training_rate):int(len_files * (training_rate + validation_rate))]
    test_files = files[int(len_files * (training_rate + validation_rate)):int(
        len_files * (training_rate + validation_rate + test_rate))]

    number_neg_pos = [0, 0]

    for file in training_files:
        number_neg_pos[0] += file_negative_positive[file][0]
        number_neg_pos[1] += file_negative_positive[file][1]
    weight_for_0 = (1 / number_neg_pos[0]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)
    weight_for_1 = (1 / number_neg_pos[1]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    training_images = []
    training_labels = []

    test_images = []
    test_labels = []

    validation_images = []
    validation_labels = []
    for file in training_files:
        training_images.extend(file_inputs[file][0])
        training_labels.extend(file_inputs[file][1])

    for file in test_files:
        test_images.extend(file_inputs[file][0])
        test_labels.extend(file_inputs[file][1])

    for file in validation_files:
        validation_images.extend(file_inputs[file][0])
        validation_labels.extend(file_inputs[file][1])

    training_images, training_labels = training_images[::every_nth], training_labels[::every_nth]
    validation_images, validation_labels = validation_images[::every_nth], validation_labels[::every_nth]
    test_images, test_labels = test_images[::every_nth], test_labels[::every_nth]

    training_images, training_labels = np.array(training_images), np.array(training_labels)
    validation_images, validation_labels = np.array(validation_images), np.array(validation_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    order = {"training_files": training_files, "validation_files": validation_files, "test_files": test_files}
    return training_images, training_labels, validation_images, validation_labels, test_images, test_labels, class_weight, order


def load_genki_from_file_with_shuffle(save_path, dimension, training_rate, validation_rate, test_rate,
                                      shuffle_order=[]):
    file = open(save_path + 'genki_data_' + str(dimension) + '.pkl', 'rb')
    images = pickle.load(file)
    labels = pickle.load(file)
    file.close()
    length = len(images)
    order = np.arange(0, length)
    np.random.shuffle(order)
    # print(type(shuffle_order))
    if (len(shuffle_order) > 0):
        order = shuffle_order

    training_images = []
    training_labels = []
    validation_images = []
    validation_labels = []
    test_images = []
    test_labels = []

    for i in range(int(length * training_rate)):
        training_images.append(images[order[i]])
        training_labels.append(labels[order[i]])

    for i in range(int(length * training_rate), int(length * (training_rate + validation_rate))):
        validation_images.append(images[order[i]])
        validation_labels.append(labels[order[i]])

    for i in range(int(length * (training_rate + validation_rate)),
                   int(length * (training_rate + validation_rate + test_rate))):
        test_images.append(images[order[i]])
        test_labels.append(labels[order[i]])

    training_images, training_labels = np.array(training_images), np.array(training_labels)
    validation_images, validation_labels = np.array(validation_images), np.array(validation_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    number_neg_pos = [length - np.sum(training_labels), np.sum(training_labels)]
    weight_for_0 = (1 / number_neg_pos[0]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)
    weight_for_1 = (1 / number_neg_pos[1]) * ((number_neg_pos[0] + number_neg_pos[1]) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    return training_images, training_labels, validation_images, validation_labels, test_images, test_labels, class_weight, order


def load_autoencoder_dataset_from_file(save_path, dimension, training_rate, validation_rate, test_rate):
    file = open(save_path + "unlabeled_data_" + str(dimension) + '.pkl', 'rb')
    images = pickle.load(file)
    file.close()

    np.random.shuffle(images)
    len_images = len(images)

    training_images = np.asarray(images[:int(len_images * training_rate)])
    validation_images = np.asarray(
        images[int(len_images * training_rate):int(len_images * (training_rate + validation_rate))])
    test_images = np.asarray(images[int(len_images * (training_rate + validation_rate)):int(
        len_images * (training_rate + validation_rate + test_rate))])

    return training_images, validation_images, test_images


def get_dictionary_combination_len(dictionary):
    parameters_len = 1
    for key, value in dictionary.items():
        parameters_len *= len(value)
    return parameters_len


def get_dictionary_combination(dictionary, idx):
    result_dict = dict()
    for key, value in dictionary.items():
        result_dict[key] = value[idx % len(value)]
        idx = (idx - idx % len(value)) // len(value)
    return result_dict


def get_confusion_matrix(truth, prediction):
    confusion_matrix = np.zeros((2, 2))
    prediction = prediction.argmax(axis=1)
    for i in range(len(prediction)):
        confusion_matrix[truth[i], prediction[i]] += 1
    return confusion_matrix
