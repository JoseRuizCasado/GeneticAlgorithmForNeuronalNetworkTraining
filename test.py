from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

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
model = Sequential()
layer1 = Dense(16, input_dim=20, activation='relu')
layer2 = Dense(12, activation='relu')
layer3 = Dense(4, activation='softmax')
# Layers are added to the model
model.add(layer1)
model.add(layer2)
model.add(layer3)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=64)
print(history)

