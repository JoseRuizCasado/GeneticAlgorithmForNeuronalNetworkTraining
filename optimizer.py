from functools import reduce
from operator import add
import random
from network import Network


class Optimizer:
    """
    Implementation of Genetic Algorithm for neuronal network optimization
    """

    def __init__(self, nn_param_choices, retain=0.4, random_select=0.1, mutate_chance=0.2):
        """
        Initialization of the Optimizer
        :param nn_param_choices: Possible network parameters
        :param retain: Percentage of population to retain after each generation
        :param random_select: Probability of a rejected network remaining in the population
        :param mutate_chance: Probability a network will be randomly mutated
        """

        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, population_size):
        """
        Create a population of random networks
        :param population_size: Number of networks to generate, the size of the population
        :return: list of Population of network objects
        """
        population = []
        for _ in range(0, population_size):
            # Create random network
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to the population
            population.append(network)

        return population

    @staticmethod
    def fitness(network):
        """
        The fitness function is the accuracy
        :param network: Network to compute fitness
        :return: network accuracy
        """
        return network.accuracy

    def crossover(self, mother, father):
        """
        Make a children as parts of their parents
        :param mother: dict with network parameters
        :param father: dict with network parameters
        :return: list with children networks
        """
        children = []
        for _ in range(2):
            child = {}

            # Pick parameters for the children
            for parameter in self.nn_param_choices:
                child[parameter] = random.choice([mother.network[parameter], father.network[parameter]])

            # Create Network object
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):
        """
        Randomly mutate one part of the network
        :param network: the network parameters to mutate
        :return: the mutated Network object
        """
        # Choose a random key
        mutation = random.choice(list(self.nn_param_choices.keys()))

        # Mutate one parameter
        network.network[mutation] = random.choice(self.nn_param_choices[mutation])

        return network

    def evolve(self, population):
        """
        :param population: a list of network parameters
        :return: the evolved population of networks
        """
        # Get scores for each network
        graded = [(self.fitness(network), network) for network in population]

        # Sort by scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the networks we keep for the next generation
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep
        parents = graded[:retain_length]

        # Randomly keep some of the refused
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find the spots we have left to fill
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        # Add children, which are bred from two remaining networks
        while len(children) < desired_length:

            # Get random parents
            father = random.randint(0, parents_length - 1)
            mother = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network
            if father != mother:
                father = parents[father]
                mother = parents[mother]

                babies = self.crossover(father, mother)

                for baby in babies:
                    # Don't grow larger tan desired length
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
