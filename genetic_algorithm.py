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
            layer1 = Dense(16, input_dim=20, activation='relu')
            layer2 = Dense(12, activation='relu')
            layer3 = Dense(4, activation='softmax')
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

    def forward_propagation(self, X_train, y_train):
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
def mutation(weights):
    """
    Mutate gen with mutator factor of 0.5
    :param weights: weights to be mutated
    :return:
    """
    selection = random.randint(0, len(weights) - 1)
    mut = random.uniform(0, 1)
    if mut >= .5:
        weights[selection] *= random.uniform(-10, 10)


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
        split = random.randint(0, np.shape(nn1_weights[i])[1]-1)
        # Iterate through after a single point and set the remaining cols to nn_2
        for j in range(split, np.shape(nn1_weights[i])[1]-1):
            nn1_weights[i][:, j] = nn2_weights[i][:, j]

        # After crossover add weights to child
        child_weights.append(nn1_weights[i])

    # Create and return child object
    child = GeneticNeuralNetwork(child_weights)
    return child


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
networks = []
pool = []
# Track Generations
generation = 0
population_size = 20
epochs = 100

# Generate n randomly weighted neural networks
for i in range(0, population_size):
    networks.append(GeneticNeuralNetwork())

# Cache Max Fitness
max_fitness = 0

# Max Fitness Weights
optimal_weights = []

# Evolution Loop
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    # Forward propagate the neural networks to compute a fitness score
    for nn in networks:
        # Propagate to calculate fitness score
        nn.forward_propagation(X_train, y_train)
        # Add to pool after calculating fitness
        pool.append(nn)

    # Clear for propagation of next children
    networks.clear()

    # Sort based on fitness
    pool = sorted(pool, key=lambda x: x.fitness)
    pool.reverse()

    # Find Max Fitness and Log Associated Weights
    for i in range(0, len(pool)):
        # If there is a new max fitness among the population
        if pool[i].fitness > max_fitness:
            max_fitness = pool[i].fitness
            print('Max Fitness: ', max_fitness)
            # Reset optimal_weights
            optimal_weights = []
            # Iterate through layers, get weights, and append to optimal
            for layer in pool[i].layers:
                optimal_weights.append(layer.get_weights()[0])
            print(optimal_weights)

    # Crossover, top 5 randomly select 2 partners for child
    for i in range(0, 5):
        for j in range(0, 2):
            # Create a child and add to networks
            temp = crossover(pool[i], random.choice(pool))
            # Add to networks to calculate fitness score next iteration
            networks.append(temp)

# Create a Genetic Neural Network with optimal initial weights
gnn = GeneticNeuralNetwork(optimal_weights)
gnn.compile_train(10)

# Test the Genetic Neural Network Out of Sample
y_hat = gnn.predict(X_test)
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_hat.round()))
