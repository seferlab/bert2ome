import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils

x = np.load("SEMBEDDINGSX.npy")
y = np.load("SEMBEDDINGSY.npy")

import tensorflow as tf

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Human 1
#xtest = np.load("RNAEMBEDDINGSX.npy")
#ytest = np.load("RNAEMBEDDINGSY.npy")

# M. musculus
#xtest = np.load("MEMBEDDINGSX.npy")
#ytest = np.load("MEMBEDDINGSY.npy")

# Human 2
xtest = np.load("DNAEMBEDDINGSX.npy")
ytest = np.load("DNAEMBEDDINGSY.npy")

def CNN_2D():
    model = Sequential()

    # layer 1
    model.add(Conv2D(32, 3, 3, input_shape=(768, 43, 1), activation="relu"))  # 16,3,3  # 1 Layer ekle bak
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
    return model

model2 = CNN_2D()


history2_ = model2.fit(np.asarray(xtrain).reshape(len(np.asarray(xtrain)),768,43,1), utils.to_categorical(ytrain,2),
                    validation_data=(np.asarray(xtest).reshape(len(np.asarray(xtest)),768,43,1), utils.to_categorical(ytest,2)),
                    epochs=20, batch_size=20, verbose=1)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_S"

plt.clf()
plt.plot(history2_.history["accuracy"])
plt.plot(history2_.history["val_accuracy"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("2D CNN Model Accuracy", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNAccuracy.jpg", dpi = 600)
print("---BERT + 2D CNN Accuracy plot was saved.")

plt.clf()
plt.plot(history2_.history["loss"])
plt.plot(history2_.history["val_loss"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("Loss", fontsize = 15)
plt.title("2D CNN Model Loss", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNLoss.jpg", dpi = 600)
print("---BERT + 2D CNN Loss plot was saved.")

print(model2.summary())

plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
cnn_predictions2 = model2.predict(np.asarray(xtest).reshape(len(np.asarray(xtest)),768,43,1))
cnn_predictions2 = np.argmax(cnn_predictions2, axis = 1)
confusion_matrix = confusion_matrix(ytest, cnn_predictions2)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 2D CNN Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+2DCNNConfusionMatrix.jpg", dpi = 600)
print("--- BERT + 2D CNN confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_cnn2, tpr_keras_bert_cnn2, thresholds_keras = roc_curve(ytest, cnn_predictions2)

from sklearn.metrics import auc
auc_keras_bert_cnn2 = auc(fpr_keras_bert_cnn2, tpr_keras_bert_cnn2)
print("AUC Score", auc_keras_bert_cnn2)

plt.clf()
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_keras_bert_cnn2, tpr_keras_bert_cnn2, label="AUC (area = {:.3f})".format(auc_keras_bert_cnn2))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 2D CNN", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNROCCurve.jpg", dpi = 600)
print("---BERT + 2D CNN ROC Curve plot was saved.")

print("----------------------------------------------")