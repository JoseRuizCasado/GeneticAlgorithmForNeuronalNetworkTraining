import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from train import compile_model, get_data


def mutation(weights):
    """
    Mutate Gen with a mutation factor of 0.05
    :param weights: list with the weights of a neuronal networks
    :return:
    """
    selection = random.randint(0, len(weights) - 1)
    mutator_factor = random.uniform(0, 1)
    if mutator_factor > .05:
        weights[selection] *= random.randint(2, 5)


def crossover(nn1, nn2):
    """
    Generate a child with two given neuronal networks
    :param nn1: Keras model
    :param nn2: Keras model
    :return: child weights list
    """
    nn1_weights = []
    nn2_weights = []
    child_weights = []

    for layer in nn1.layers:
        nn1_weights.append(layer.get_weights()[0])

    for layer in nn2.layers:
        nn2_weights.append(layer.get_weights()[0])

    for i in range(len(nn1_weights)):
        split = random.randint(0, np.shape(nn1.weights[i])[1] - 1)
        # Iterate through after a single point and set the remaining cols to nn_2
        for j in range(split, np.shape(nn1_weights[i])[1] - 1):
            nn1_weights[i][:, j] = nn2_weights[i][:, j]

        child_weights.append(nn1_weights[i])

    return child_weights


def fitness(model, X_train, y_train):
    """
    Apply the fitness function to a given neuronal network
    :param model: Keras model to apply the fitness function
    :param X_train: train features
    :param y_train: train labels
    :return: fitness value of the neuronal network
    """
    y_hat = model.predict(X_train)
    return accuracy_score(y_train, y_hat.round())


def genetic_train(network, epochs, population_size):
    """
    Apply genetic algorithm to train a given neuronal network
    :param network: neuronal network architecture.
    :param epochs: number of epochs to be performed
    :param population_size: size of the populations of
    :return:
    """
    n_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_data()
    population = []
    for _ in range(population_size):
        population.append(compile_model(network, n_classes, input_shape, genetic_flag=True))

    performed_epochs = 0
    while epochs > performed_epochs:
        performed_epochs += 1
        print(f'Generation {performed_epochs}')
        fitness_values = []
        for nn in population:
            fitness_values.append({
                'model': nn,
                'fitness': fitness(nn, x_train, y_train)
            })

        fitness_values = sorted(fitness_values, reverse=True, key=lambda x: x['fitness'])
        print(fitness_values)
