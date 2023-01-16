import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn import metrics
import tensorflow as tf


x = np.load("SEMBEDDINGSX.npy")
y = np.load("SEMBEDDINGSY.npy")


acc_per_fold = []
loss_per_fold = []

num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
for train, test in kfold.split(x, y):
    model = Sequential()

    # layer 1
    model.add(Conv2D(32, 3, 3, input_shape=(768, 43, 1), activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.1))

    # layer 2
    model.add(Conv2D(16, 3, 3, activation="relu"))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.2))

    # Flattening Layer:
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))

    # Last Layer:
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy", "mse", "mape", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history2_ = model.fit(np.asarray(x[train]).reshape(len(np.asarray(x[train])),768,43,1), utils.to_categorical(y[train],2),
                epochs=20, batch_size=20, verbose=1)

    scores = model.evaluate(np.asarray(x[test]).reshape(len(np.asarray(x[test])), 768, 43, 1), utils.to_categorical(y[test], 2))
    print(scores[0])
    print(scores[1])

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    fold_no = fold_no + 1

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for 5 folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')