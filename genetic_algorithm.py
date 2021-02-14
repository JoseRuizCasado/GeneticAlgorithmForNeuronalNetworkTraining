import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class GeneticNeuralNetwork(Sequential):
    """
    Generalization of Keras Sequential model
    """

    def __init__(self, child_weights=None):
        """
        Initialize Sequential Model and calling super class
        :param child_weights: list with weights to be set as network weights
        """
        super().__init__()
        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are created and randomly generated
            layer1 = Dense(16, input_dim=20, activation='relu', kernel_initializer='random_normal')
            layer2 = Dense(12, activation='relu', kernel_initializer='random_normal')
            layer3 = Dense(4, activation='softmax', kernel_initializer='random_normal')
            # Layers are added to the model
            self.add(layer1)
            self.add(layer2)
            self.add(layer3)
        # If weights are provided set them within the layers
        else:
            # Set weights within the layers
            self.add(
                Dense(
                    16,
                    input_dim=20,
                    activation='relu',
                    weights=[child_weights[0], np.zeros(16)])
            )
            self.add(
                Dense(
                    12,
                    activation='relu',
                    weights=[child_weights[1], np.zeros(12)])
            )
            self.add(
                Dense(
                    4,
                    activation='softmax',
                    weights=[child_weights[2], np.zeros(4)])
            )

    def fitness_function(self, X_train, y_train):
        """
        Compute the fitness of the network
        :param X_train:
        :param y_train:
        :return:
        """
        # Forward propagation
        y_predicted = self.predict(X_train)
        # Compute fitness score
        self.fitness = accuracy_score(y_train, y_predicted.round())

    def compile_train(self, epochs):
        """
        Standard backpropagtion method
        :param epochs: number of epochs to perform the training
        :return:
        """
        self.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.fit(X_train, y_train, epochs=epochs)


# Chance to mutate weights
def mutation(nn):
    """
    Mutate gen with mutator factor of 0.5
    :param nn: Neuronal network to perform the mutation
    :return: mutate neuronal network
    """
    mutate_weights = []
    for l in nn.layers:
        weights = l.get_weights()[0]
        selection = random.randint(0, len(weights) - 1)
        mut = random.uniform(0, 1)
        if mut >= .5:
            weights[selection] *= random.uniform(-10, 10)
        mutate_weights.append(weights)

    mutate = GeneticNeuralNetwork(mutate_weights)
    return mutate


# Crossover traits between two Genetic Neural Networks
def crossover(nn1, nn2):
    """
    Create a child network with two given networks
    :param nn1: GeneticNeuronalNetwork instance
    :param nn2: GeneticNeuronalNetwork instance
    :return: child GeneticNeuronalNetwork instance
    """

    nn1_weights = []
    nn2_weights = []
    child_weights = []
    # Get all weights from the layers in the both networks
    for layer in nn1.layers:
        nn1_weights.append(layer.get_weights()[0])

    for layer in nn2.layers:
        nn2_weights.append(layer.get_weights()[0])

    # Iterate through all weights for crossover
    for i in range(0, len(nn1_weights)):
        # Get single point to split the matrix in parents based on # of cols
        split = random.randint(0, np.shape(nn1_weights[i])[1] - 1)
        # Iterate through after a single point and set the remaining cols to nn_2
        for j in range(split, np.shape(nn1_weights[i])[1] - 1):
            nn1_weights[i][:, j] = nn2_weights[i][:, j]

        # After crossover add weights to child
        child_weights.append(nn1_weights[i])

    # Create and return child object
    child = GeneticNeuralNetwork(child_weights)
    return child


def selection_function(nn_list, listed_fitness):
    """
    Select two parent based on the selection function: fitness_i/sum(fitness_i)
    :param nn_list: list with all the neuronal network of the actual population
    :param listed_fitness: list with the fitness of all the neuronal networks
    :return: parents to apply crossover
    """
    fitness_sum = sum(listed_fitness)
    selection_probability = [f / fitness_sum for f in listed_fitness]
    return random.choices(nn_list, weights=selection_probability, k=4)


train_dataset = pd.read_csv('./data/train.csv')

# Transform pandas dataframe into numpy array
X = train_dataset.iloc[:, :20].values
y = train_dataset.iloc[:, 20:21].values

# Normalizing the data
sc = StandardScaler()
X = sc.fit_transform(X)

# Transform y to One hot encode
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Create a List of all active GeneticNeuralNetworks
population = []
# Track Generations
generation = 0
population_size = 20
epochs = 100

# Generate n randomly weighted neural networks
for i in range(0, population_size):
    population.append(GeneticNeuralNetwork())

# Cache Max Fitness
max_fitness = 0

# Max Fitness Weights
optimal_weights = []

# Evolution Loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    # Forward propagate the neural networks to compute a fitness score
    fitness_list = []
    new_generation = []
    for nn in population:
        # Propagate to calculate fitness score
        nn.fitness_function(X_train, y_train)
        # Add to pool after calculating fitness
        new_generation.append(nn)
        fitness_list.append(nn.fitness)

    # Sort based on fitness
    new_generation = sorted(new_generation, key=lambda x: x.fitness)
    new_generation.reverse()
    print(f'Max fitness: {new_generation[0].fitness}')
    fitness_list.sort(reverse=True)
    # print(f'Pop: {new_generation[0].fitness} list {fitness_list[0]}')

    # Select 4 parents
    father1, mother1, father2, mother2 = selection_function(new_generation, fitness_list)
    # print(f'Father1 fitness: {father1.fitness}, Mother1 fitness: {mother1.fitness}')
    # print(f'Father2 fitness: {father2.fitness}, Mother2 fitness: {mother2.fitness}')
    child1 = crossover(father1, mother1)
    child1.fitness_function(X_train, y_train)
    child2 = crossover(father2, mother2)
    child2.fitness_function(X_train, y_train)
    # Replace 2 nn with worst fitness with child, if better
    new_generation.append(child1)
    new_generation.append(child2)
    new_generation = sorted(new_generation, key=lambda x: x.fitness)
    new_generation.reverse()
    del new_generation[population_size-2:population_size]

    population = []
    # Mutate the generation and keep if better
    for gen in new_generation:
        new_gen = mutation(gen)
        new_gen.fitness_function(X_train, y_train)
        if new_gen.fitness > gen.fitness:
            population.append(new_gen)
        else:
            population.append(gen)



# Create a Genetic Neural Network with optimal initial weights
gnn = GeneticNeuralNetwork(optimal_weights)
gnn.compile_train(10)

# Test the Genetic Neural Network Out of Sample
y_hat = gnn.predict(X_test)
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_hat.round()))
