##
import pickle
import tensorflow as tf

import FNN
import CNN
#import RNN # works only with Tensorflow 1.9
import TRANSFER
import AUTOENCODER
import AUTOENCODER_CLASSIFIER
import helping_functions as hf
import input_generation.amfed_generator as amfed_generator
import input_generation.genki_generator as genki_generator
import input_generation.unlabeled_generator as unlabeled_generator


##
original_genki_dataset_path = r""
dlib_shape_predictor_path = r""
original_amfed_dataset_path = r""
unlabeled_files = []
save_path = r""
dataset_path = r""

image_size = 224

amfed_generator.generate_dataset(image_size, original_amfed_dataset_path, dataset_path)
genki_generator.generate_dataset(image_size, original_amfed_dataset_path, dlib_shape_predictor_path, dataset_path)
unlabeled_generator.generate_dataset(image_size, unlabeled_files, dlib_shape_predictor_path, dataset_path)

##


# FNN examine: number of layers, number of neurons
parameters = {"layer_nb": [1, 2, 3, 4, 5, 6], "layer_size": [10, 20, 30, 40, 50], "dropout": [0], "activation": ["relu", "relu"], "contrast": [0], "flip": [False],
              "translation_x": [0], "translation_y": [0], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(10):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        FNN.train_FNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# FNN examine: dropouts, activation functions
parameters = {"layer_nb": [2, 3], "layer_size": [20, 30], "dropout": [0, 0.2], "activation": [["relu", "relu"], ["tanh", "tanh"], ["sigmoid", "sigmoid"], [tf.keras.layers.LeakyReLU(alpha=0.1), "leaky_relu"]], "contrast": [0], "flip": [False],
              "translation_x": [0], "translation_y": [0], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(3):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        FNN.train_FNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# FNN examine: random contrast, random flip
parameters = {"layer_nb": [3], "layer_size": [30], "dropout": [0], "activation": [["relu", "relu"]], "contrast": [0, 0.5], "flip": [False, True],
              "translation_x": [0], "translation_y": [0], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(30):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        FNN.train_FNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# FNN examine: random x translation, random y translation
parameters = {"layer_nb": [3], "layer_size": [30], "dropout": [0], "activation": [["relu", "relu"]], "contrast": [0], "flip": [True],
              "translation_x": [0, 0.05, 0.1, 0.25], "translation_y": [0, 0.05, 0.1, 0.25], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(3):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        FNN.train_FNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# FNN amfed examine: number of layers, number of neurons
parameters = {"layer_nb": [2, 3], "layer_size": [20, 30], "dropout": [0], "activation": ["relu", "relu"], "contrast": [0], "flip": [False],
              "translation_x": [0], "translation_y": [0], "dataset": ["amfed"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(5):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        FNN.train_FNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

## CNN examine: number of kernels, number of convolutional layers, kernel sizes
parameters = {"layer_nb": [1, 2, 3, 4], "kernel_nb": [16, 32], "kernel_size": [3, 5, 7, 9], "dropout": [0], "contrast": [0], "flip": [False],
              "translation_x": [0], "translation_y": [0], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(5):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        CNN.train_CNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# CNN examine: random contrast, random flip
parameters = {"layer_nb": [3], "kernel_nb": [32], "kernel_size": [5], "dropout": [0], "contrast": [0, 0.25, 0.5, 0.75], "flip": [False, True],
              "translation_x": [0], "translation_y": [0], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(10):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        CNN.train_CNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# CNN examine: random x translation, random y translation
parameters = {"layer_nb": [3], "kernel_nb": [32], "kernel_size": [5], "dropout": [0], "contrast": [0.5], "flip": [True],
              "translation_x": [0, 0.05, 0.1, 0.25], "translation_y": [0, 0.05, 0.1, 0.25], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(5):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        CNN.train_CNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# CNN examine: dropout
parameters = {"layer_nb": [3], "kernel_nb": [32], "kernel_size": [5], "dropout": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], "contrast": [0.5], "flip": [True],
              "translation_x": [0.05], "translation_y": [0.05], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(5):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        CNN.train_CNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# CNN amfed examine: number of convolutional layers, kernel sizes
parameters = {"layer_nb": [3, 4], "kernel_nb": [32], "kernel_size": [5, 7], "dropout": [0], "contrast": [0.5], "flip": [True],
              "translation_x": [0.05], "translation_y": [0.05], "dataset": ["amfed"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(3):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        CNN.train_CNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

##TRANSFER examine: translation, model
parameters = {"model": ["xception", "resnet"], "contrast": [0], "flip": [True], "translation": [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(3):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        TRANSFER.train_TRANSFER(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# TRANSFER examine: contrast
parameters = {"model": ["resnet"], "contrast": [0, 0.25, 0.5, 0.75], "flip": [True], "translation": [0.3], "dataset": ["genki"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(3):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        TRANSFER.train_TRANSFER(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

# TRANSFER amfed examine
parameters = {"model": ["resnet"], "contrast": [0.75], "flip": [True], "translation": [0.3], "dataset": ["amfed"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(3):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        TRANSFER.train_TRANSFER(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

##RNN examine: freezing, sequence length, cell type (Tensorflow 1.9 needed)
# parameters = {"cnn_model_path": [""], "seq_len": [3, 5, 10], "cell": [[tf.keras.layers.SimpleRNN, "SimpleRNN"], [tf.keras.layers.LSTM, "LSTM"], [tf.keras.layers.GRU, "GRU"]], "freeze": [False, True]}
# combination_len = hf.get_dictionary_combination_len(parameters)
# for train_number in range(3):
#     for i in range(combination_len):
#         call_params = hf.get_dictionary_combination(parameters, i)
#         RNN.train_RNN(dataset_path=dataset_path, image_size=image_size, save_path=save_path + str(train_number) + "_", **call_params)

##AUTOENCODER examine: latent dimension
parameters = {"latent_dimension": [30, 60, 90, 120, 150, 200]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(5):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        AUTOENCODER.train_AUTOENCODER(dataset_path=dataset_path, save_path=save_path + str(train_number) + "_", **call_params)

##AUTOENCODER_CLASSIFIER examine: latent division, dataset
parameters = {"latent_dimension": [30, 60, 90, 120, 150, 200], "dataset": ["genki", "amfed"]}
combination_len = hf.get_dictionary_combination_len(parameters)
for train_number in range(5):
    for i in range(combination_len):
        call_params = hf.get_dictionary_combination(parameters, i)
        call_params["model_path"] = AUTOENCODER.get_full_save_path(save_path + str(0) + "_", **call_params) + "_model.hdf5"
        AUTOENCODER_CLASSIFIER.train_AUTOENCODER_CLASSIFIER(dataset_path=dataset_path, save_path=save_path, **call_params)
##

str = AUTOENCODER_CLASSIFIER.get_full_save_path(save_path, **call_params)
file = open(str + "_history.pkl", "rb")
history = pickle.load(file)
confusion = pickle.load(file)
order = pickle.load(file)
file.close()
##
