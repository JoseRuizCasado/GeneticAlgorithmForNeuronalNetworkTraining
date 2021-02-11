from optimizer import Optimizer
from tqdm import tqdm


def train_networks(population, genetic_train=False):
    """
    Train each network
    :param population: Current population of networks
    :param genetic_train: flag to use or not genetic algorithm for training
    """
    pbar = tqdm(total=len(population))
    for network in population:
        network.train(genetic_train)
        pbar.update(1)

    pbar.close()


def get_average_accuracy(population):
    """
    Get the average accuracy for a group of networks
    :param population: list of networks
    :return: the average accuracy of a population of networks
    """
    total_accuracy = 0
    for network in population:
        total_accuracy += network.accuracy

    return total_accuracy / len(population)


def generate(generations, population, nn_param_choices, genetic_train=False):
    """
    Generate a network with the genetic algorithm.
    :param generations: Number of times to evole the population
    :param population: Number of networks in each generation
    :param nn_param_choices: Parameter choices for networks
    :param genetic_train: flag to use or not genetic algorithm for training
    """
    optimizer = Optimizer(nn_param_choices=nn_param_choices)
    networks = optimizer.create_population(population)

    # Evolve the generation
    for i in range(generations):
        print(f'***Doing generation {i} of {generations}***')

        # Train and get accuracy for networks
        train_networks(networks, genetic_train)

        # Get the average accuracy for current generation
        average_accuracy = get_average_accuracy(networks)
        print(f'Generation average accuracy: {average_accuracy * 100}%')

        # Evolve, except on the last iteration
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    # Sort the final population
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    for network in networks[:5]:
        network.print_network()


def main():
    generations = 10
    population = 20

    nn_param_choices = {
        'nb_neurons': [64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }
    print(f'***Evolving {generations} generations with population {population}***')
    generate(generations=generations, population=population, nn_param_choices=nn_param_choices, genetic_train=True)


if __name__ == '__main__':
    main()
