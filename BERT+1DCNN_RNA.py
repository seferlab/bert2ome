import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Activation, Conv1D, ZeroPadding1D, MaxPooling1D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam

x = np.load("RNAEMBEDDINGSX.npy")
y = np.load("RNAEMBEDDINGSY.npy")

import tensorflow as tf

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.10, random_state = 42)

def CNN_1D():
    model = Sequential()

    # layer 1
    model.add(Conv1D(32, 3, input_shape=(43 * 768, 1), activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.1))

    # layer 2
    model.add(Conv1D(16, 3, activation="relu"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    # Flattening Layer:
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))

    # Last Layer:
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy", "mse", "mape", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

model1 = CNN_1D()

history1_ = model1.fit(np.asarray(xtrain).reshape(len(np.asarray(xtrain)),43*768,1), utils.to_categorical(ytrain,2),
                    validation_data=(np.asarray(xtest).reshape(len(np.asarray(xtest)),43*768,1), utils.to_categorical(ytest,2)),
                    epochs=15, batch_size=20, verbose=1)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_RNA"


plt.clf()
plt.plot(history1_.history["accuracy"])
plt.plot(history1_.history["val_accuracy"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("1D CNN Model Accuracy", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+1DCNNAccuracy.jpg", dpi = 600)
print("---BERT + 1D CNN Accuracy plot was saved.")


plt.clf()
plt.plot(history1_.history["loss"])
plt.plot(history1_.history["val_loss"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("Loss", fontsize = 15)
plt.title("1D CNN Model Loss", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+1DCNNLoss.jpg", dpi = 600)
print("---BERT + 1D CNN Loss plot was saved.")

print(model1.summary())

plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
cnn_predictions1 = model1.predict(xtest)
cnn_predictions1 = np.argmax(cnn_predictions1, axis = 1)
confusion_matrix = confusion_matrix(ytest, cnn_predictions1)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 1D CNN Confusion Matrix", fontsize = 20)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_RNA"

plt.savefig(directory + "/BERT+1DCNNConfusionMatrix.jpg", dpi = 600)
print("--- BERT + 1D CNN confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_cnn1, tpr_keras_bert_cnn1, thresholds_keras_bert_cnn1 = roc_curve(ytest, cnn_predictions1)

from sklearn.metrics import auc
auc_keras_bert_cnn1 = auc(fpr_keras_bert_cnn1, tpr_keras_bert_cnn1)
print("AUC Score", auc_keras_bert_cnn1)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_cnn1, tpr_keras_bert_cnn1, label='AUC (area = {:.3f})'.format(auc_keras_bert_cnn1))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 1D CNN", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+1DCNNROCCurve.jpg", dpi = 600)
print("---BERT + 1D CNN ROC Curve plot was saved.")

print("----------------------------------------------")
