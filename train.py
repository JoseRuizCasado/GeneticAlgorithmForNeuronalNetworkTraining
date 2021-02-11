from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)


def get_data():
    """
    Load data from data folder
    :return: number of classes, batch size, input_shape, x_train, x_test, y_train, y_test
    """

    n_classes = 4
    batch_size = 64
    input_shape = 20

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

    return n_classes, batch_size, input_shape, X_train, X_test, y_train, y_test


def compile_model(network, n_classes=4, input_shape=20, genetic_flag=False):
    """
    Compile sequential model with network hyperparameters.
    :param network: object with info about hyperparameters
    :param n_classes: number of classes in dataset, needed for output layer
    :param input_shape: shape of input of the neuronal network
    :param genetic_flag: flag to compile or not the model
    :return: compiled model
    """
    n_layers = network['nb_layers']
    n_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer to the model
    for l in range(n_layers):
        # First layer is the input layer
        if l == 0:
            model.add(Dense(n_neurons, activation=activation, input_dim=input_shape))
        else:
            model.add(Dense(n_neurons, activation=activation))

        model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(n_classes, activation='softmax'))

    if not genetic_flag:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_and_score(network):
    n_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_data()
    model = compile_model(network, n_classes, input_shape)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, verbose=0, validation_data=(x_test, y_test), 
              callbacks=[early_stopper])
    score = model.evaluate(x_test, y_test, verbose=0)
    # Use accuracy
    return score[1]
