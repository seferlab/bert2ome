#importing tensorflow
import tensorflow as tf
#importing keras from tensorflow
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
#importing random search
from kerastuner import RandomSearch
from tensorflow.keras import utils
import matplotlib.pyplot as plt


x = np.load("RNAEMBEDDINGSX.npy")
y = np.load("RNAEMBEDDINGSY.npy")


def build_model(hp):
    # create model object
    model = keras.Sequential([

        # adding first convolutional layer
        keras.layers.Conv2D(
            # adding filter
            filters=hp.Int('conv_1_filter', min_value=10, max_value=32, step=4),
            # adding filter size or kernel size
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 7]),
            # activation function
            activation='relu',
            input_shape=(768, 43, 1)),

        # adding second convolutional layer
        keras.layers.Conv2D(
            # adding filter
            filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=4),
            # adding filter size or kernel size
            kernel_size=hp.Choice('conv_2_kernel', values=[3, 6]),
            # activation function
            activation='relu'
        ),

        # adding flatten layer
        keras.layers.Flatten(),
        # adding dense layer
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=80, step=4),
            activation='relu'
        ),

        # output layer
        keras.layers.Dense(2, activation='softmax')
    ])

    # compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=["accuracy", "mse", "mape",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall()])

    model.summary()
    return model

#creating randomsearch object
tuner = RandomSearch(build_model,
                    objective='val_accuracy',
                    max_trials = 5, directory = "output", project_name = "AfterHyperParameterTuning")

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.10, random_state = 42)


tuner.search(np.asarray(xtrain).reshape(len(np.asarray(xtrain)),768,43,1),
              utils.to_categorical(ytrain,2),
              epochs = 8,
              validation_split = 0.2)

model_2D_ht = tuner.get_best_models(num_models=1)[0]

print(model_2D_ht.summary())

history3 = model_2D_ht.fit(np.asarray(xtest).reshape(len(np.asarray(xtest)),768,43,1), utils.to_categorical(ytest,2),
                           epochs=15, batch_size = 20, validation_split=0.1,
                           initial_epoch=1)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_RNA"

plt.clf()
plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("2D CNN Model Accuracy", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNAccuracywithHyp.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning Accuracy plot was saved.")

plt.clf()
plt.plot(history3.history["loss"])
plt.plot(history3.history["val_loss"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("Loss", fontsize = 15)
plt.title("2D CNN Model Loss", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNLosswithHyp.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning Loss plot was saved.")

print(model_2D_ht.summary())

plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
cnn_predictions3 = model_2D_ht.predict(np.asarray(xtest).reshape(len(np.asarray(xtest)),768,43,1))
cnn_predictions3 = np.argmax(cnn_predictions3, axis = 1)
confusion_matrix = confusion_matrix(ytest, cnn_predictions3)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 2D CNN + H. Tuning Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+2DCNNConfusionMatrixwithHyp.jpg", dpi = 600)
print("--- BERT + 2D CNN + Hyperparameter Tuning confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp, thresholds_keras = roc_curve(ytest, cnn_predictions3)

from sklearn.metrics import auc
auc_keras_bert_cnn2_hyp = auc(fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp)
print("AUC Score", auc_keras_bert_cnn2_hyp)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp, label='Keras (area = {:.3f})'.format(auc_keras_bert_cnn2_hyp))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 2D CNN", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+2DCNNROCCurvewithHyp.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning ROC Curve plot was saved.")

print("----------------------------------------------")

