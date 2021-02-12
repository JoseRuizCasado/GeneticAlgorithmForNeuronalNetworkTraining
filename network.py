import random
from train import train_and_score


class Network:
    """
    Object to represent a Neuronal Network Gene of the algorithm
    """

    def __init__(self, nn_param_choices):
        """
        Initialize the network
        :param nn_param_choices: Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0
        self.nn_param_choices = nn_param_choices
        self.network = {}

    def create_random(self):
        """
        Create a random network.
        """
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """
        Set network properties.
        :param network: Network properties to be set.
        """
        self.network = network

    def train(self, genetic_train_flag=False):
        """
        Train network and get the accuracy.
        :param genetic_train_flag: flag to use or not genetic algorithm for training
        """
        if self.accuracy == 0:
            self.accuracy = train_and_score(self.network)

    def print_network(self):
        """
        Print info about the network
        """
        print(self.network)
        print(f'Network accuracy: {self.accuracy * 100}')
