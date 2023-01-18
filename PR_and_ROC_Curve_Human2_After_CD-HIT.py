import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv2D
from kerastuner import RandomSearch
from sklearn.metrics import precision_recall_curve
import pandas as pd
from tensorflow.keras.layers import Dense, Activation, Conv1D, ZeroPadding1D, MaxPooling1D, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Dataset reading
rmbase_dataset_ = pd.read_excel("RMBase_800_CD-HIT.xlsx")
# print(rmbase_dataset_.shape)

y = rmbase_dataset_.label
x = rmbase_dataset_.drop(["label"], axis = 1)


def one_hot_coversion(each_row):
    nuc_vector = []
    nuc_vector = np.array(nuc_vector)

    for i in each_row:
        if i == "A":
            nuc_vector = np.concatenate((nuc_vector, [1, 0, 0, 0]))
        elif i == "G":
            nuc_vector = np.concatenate((nuc_vector, [0, 1, 0, 0]))
        elif i == "C":
            nuc_vector = np.concatenate((nuc_vector, [0, 0, 1, 0]))
        elif i == "U":
            nuc_vector = np.concatenate((nuc_vector, [0, 0, 0, 1]))

    return nuc_vector

all_vectors = []
for i in range(0, len(x)):
    row_vector = one_hot_coversion(x[0][i])
    all_vectors.append(row_vector)

new_x = pd.DataFrame(all_vectors)

print("One-hot conversion was done.")

# SMOTE Part

from collections import Counter
from imblearn.over_sampling import SMOTE
counter = Counter(y)
print(counter)

oversample = SMOTE()
new_x, y = oversample.fit_resample(new_x, y)

counter = Counter(y)
print("After SMOTE:")
print(counter)

# Model results images folder:
directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_PR_Curve_Human2 - After_CD-HIT"

################################################################################################
################################### BASE MODELS START HERE #####################################
################################################################################################

######################################## Decision Tree #########################################


dt = DecisionTreeClassifier(random_state = 0)

xtrain, xtest, ytrain, ytest = train_test_split(new_x, y, test_size = 0.10, random_state=42)


dt.fit(xtrain, ytrain)
decision_tree_predictions = dt.predict(xtest)
print(decision_tree_predictions)
dt_predictions_proba = dt.predict_proba(xtest)
print(dt_predictions_proba)
dt_predictions_proba = dt_predictions_proba[:,1]
print(dt_predictions_proba)

print("Model accuracy score: " + str(accuracy_score(decision_tree_predictions, ytest)))

f1_score = f1_score(ytest, decision_tree_predictions)
print("f1_score:", f1_score)

print("Precision:", metrics.precision_score(decision_tree_predictions, ytest))
print("Recall:", metrics.recall_score(decision_tree_predictions, ytest))

import seaborn as sns
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, decision_tree_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("Decision Tree Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/DecisionTreeConfusionMatrix.jpg", dpi = 600)
print("---Decision Tree confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_dt, tpr_keras_dt, thresholds_keras_dt = roc_curve(ytest, dt_predictions_proba)

from sklearn.metrics import auc
auc_keras_dt = auc(fpr_keras_dt, tpr_keras_dt)
print("AUC Score:", auc_keras_dt)

plt.clf()
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_keras_dt, tpr_keras_dt, label="AUC (area = {:.3f})".format(auc_keras_dt))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for Decision Tree", fontsize = 20)
plt.legend(loc='best',  prop={'size': 15})
plt.savefig(directory + "/DecisionTreeROCCurve.jpg", dpi = 600)
print("---Decision Tree ROC Curve plot was saved.")

print("----------------------------------------------")


precision_dt, recall_dt, _ = precision_recall_curve(ytest, dt_predictions_proba)
auc_precision_recall_dt = auc(recall_dt, precision_dt)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_dt, precision_dt, label="Decision Tree: {:.3f}".format(auc_precision_recall_dt))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for BERT + DT", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/DecisionTreePRCurvewithHyp.jpg", dpi = 600)
print("---Decision Tree + PR Curve plot was saved.")

print("----------------------------------------------")



######################################## Random Forest #########################################

rfc = RandomForestClassifier(random_state=0, criterion="entropy")

rfc.fit(xtrain, ytrain)
random_forest_predictions = rfc.predict(xtest)
print(random_forest_predictions)
rf_predictions_proba = rfc.predict_proba(xtest)
rf_predictions_proba = rf_predictions_proba[:,1]
print(rf_predictions_proba)

print("Model accuracy score: " + str(accuracy_score(random_forest_predictions, ytest)))

#f1_score = f1_score(ytest, random_forest_predictions)
#print("f1_score:", f1_score)

print("Precision:", metrics.precision_score(random_forest_predictions, ytest))
print("Recall:", metrics.recall_score(random_forest_predictions, ytest))

plt.clf()
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, random_forest_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("Random Forest Confusion Matrix", fontsize = 20)
plt.savefig(directory + "/RandomForestConfusionMatrix.jpg", dpi = 600)
print("---Random Forest confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_rf, tpr_keras_rf, thresholds_keras_rf = roc_curve(ytest, rf_predictions_proba)

from sklearn.metrics import auc
auc_keras_rf = auc(fpr_keras_rf, tpr_keras_rf)
print("AUC Score:", auc_keras_rf)


plt.clf()
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_keras_rf, tpr_keras_rf, label="AUC (area = {:.3f})".format(auc_keras_rf))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for Random Forest", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/RandomForestROCCurve.jpg", dpi = 600)
print("---Random Forest ROC Curve plot was saved.")

print("----------------------------------------------")


precision_rf, recall_rf, _ = precision_recall_curve(ytest, rf_predictions_proba)
auc_precision_recall_rf = auc(recall_rf, precision_rf)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_rf, precision_rf, label="Random Forest: {:.3f}".format(auc_precision_recall_rf))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for RF", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/RandomForestPRCurve.jpg", dpi = 600)
print("---Random Forest + PR Curve plot was saved.")

print("----------------------------------------------")



# ######################################### XGBoost ###########################################

from xgboost import XGBClassifier

xgmodel = XGBClassifier()

xtrain, xtest, ytrain, ytest = train_test_split(new_x, y, test_size = 0.10, random_state=42)


xgmodel.fit(xtrain, ytrain, verbose=False)

xgboost_predictions = xgmodel.predict(xtest)
xgb_predictions_proba = xgmodel.predict_proba(xtest)
xgb_predictions_proba = xgb_predictions_proba[:,1]

print("Model accuracy score: " + str(accuracy_score(xgboost_predictions, ytest)))
print("f1 score: ", metrics.f1_score(xgboost_predictions, ytest))
print("Precision:", metrics.precision_score(xgboost_predictions, ytest))
print("Recall:", metrics.recall_score(xgboost_predictions, ytest))

plt.clf()
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, xgboost_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("XGBoost Confusion Matrix", fontsize = 20)
plt.savefig(directory + "/XGBoostConfusionMatrix.jpg", dpi = 600)
print("---XGBoost confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_xgb, tpr_keras_xgb, thresholds_keras_xgb = roc_curve(ytest, xgb_predictions_proba)

from sklearn.metrics import auc
auc_keras_xgb = auc(fpr_keras_xgb, tpr_keras_xgb)
print("AUC Score:", auc_keras_xgb)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_xgb, tpr_keras_xgb, label='AUC (area = {:.3f})'.format(auc_keras_xgb))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for XGBoost", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/XGBoostROCCurve.jpg", dpi = 600)
print("---XGBoost ROC Curve plot was saved.")

print("----------------------------------------------")

precision_xgb, recall_xgb, _ = precision_recall_curve(ytest, xgb_predictions_proba)
auc_precision_recall_xgb = auc(recall_xgb, precision_xgb)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_xgb, precision_xgb, label="XGBoost: {:.3f}".format(auc_precision_recall_xgb))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for XGBoost", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/XGBoostPRCurvewithHyp.jpg", dpi = 600)
print("---XGBoost + PR Curve plot was saved.")

print("----------------------------------------------")

################################################################################################
################################### BASE MODELS END HERE #######################################
################################################################################################


# ################################### BERT + Random Forest #####################################

x = np.load("HUMAN2EMBEDDINGSX.npy")
y = np.load("HUMAN2EMBEDDINGSY.npy")

print("-------- BERT + Random Forest Model --------")

rfc = RandomForestClassifier(random_state=0)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

rfc.fit(xtrain, ytrain)
bert_rf_predictions = rfc.predict(xtest)
bert_rf_predictions_proba = rfc.predict_proba(xtest)
bert_rf_predictions_proba = bert_rf_predictions_proba[:,1]
print(bert_rf_predictions)

print("Model accuracy score: " + str(accuracy_score(bert_rf_predictions, ytest)))

#f1_score = f1_score(ytest, bert_rf_predictions)
#print("f1_score:", f1_score)

print("Precision:", metrics.precision_score(bert_rf_predictions, ytest))
print("Recall:", metrics.recall_score(bert_rf_predictions, ytest))

plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, bert_rf_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + Random Forest Confusion Matrix", fontsize = 20)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_PR_Curve_Human2 - After_CD-HIT"

plt.savefig(directory + "/BERT+RandomForestConfusionMatrix.jpg", dpi = 600)
print("--- BERT + Random Forest confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_rf, tpr_keras_bert_rf, thresholds_keras_bert_rf = roc_curve(ytest, bert_rf_predictions_proba)

from sklearn.metrics import auc
auc_keras_bert_rf = auc(fpr_keras_bert_rf, tpr_keras_bert_rf)
print("AUC Score:", auc_keras_bert_rf)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_rf, tpr_keras_bert_rf, label="BERT + Random Forest: {:.3f})".format(auc_keras_bert_rf))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + Random Forest", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+RandomForestROCCurve.jpg", dpi = 600)
print("---BERT + Random Forest ROC Curve plot was saved.")

print("----------------------------------------------")


precision_bert2_rf, recall_bert2_rf, _ = precision_recall_curve(ytest, bert_rf_predictions_proba)
auc_precision_recall_bert2_rf = auc(recall_bert2_rf, precision_bert2_rf)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_bert2_rf, precision_bert2_rf, label="BERT + Random Forest: {:.3f}".format(auc_precision_recall_bert2_rf))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for BERT + RF", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+RandomForestPRCurvewithHyp.jpg", dpi = 600)
print("---BERT + Random Forest + PR Curve plot was saved.")

print("----------------------------------------------")

########################################## BERT + XGBoost ###############################################

from xgboost import XGBClassifier
xgmodel = XGBClassifier()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.10, random_state = 42)

xgmodel.fit(xtrain, ytrain, verbose=False)

bert_xgb_predictions = xgmodel.predict(xtest)
bert_xgb_predictions_proba = xgmodel.predict_proba(xtest)
bert_xgb_predictions_proba = bert_xgb_predictions_proba[:,1]

print("Model accuracy score: " + str(accuracy_score(bert_xgb_predictions, ytest)))
print("f1 score: ", metrics.f1_score(bert_xgb_predictions, ytest))
print("Precision:", metrics.precision_score(bert_xgb_predictions, ytest))
print("Recall:", metrics.recall_score(bert_xgb_predictions, ytest))

plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, bert_xgb_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + XGBoost Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+XGBoostConfusionMatrix.jpg", dpi = 600)
print("--- BERT + XGBoost confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_xgb, tpr_keras_bert_xgb, thresholds_keras_bert_xgb = roc_curve(ytest, bert_xgb_predictions_proba)

from sklearn.metrics import auc
auc_keras_bert_xgb = auc(fpr_keras_bert_xgb, tpr_keras_bert_xgb)
print("AUC Score:", auc_keras_bert_xgb)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_xgb, tpr_keras_bert_xgb, label="XGBoost = {:.3f})".format(auc_keras_bert_xgb))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + XGBoost", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+XGBoostROCCurve.jpg", dpi = 600)
print("---BERT + XGBoost ROC Curve plot was saved.")

print("----------------------------------------------")

precision_bert2_xgb, recall_bert2_xgb, _ = precision_recall_curve(ytest, bert_xgb_predictions_proba)
auc_precision_recall_bert2_xgb = auc(recall_bert2_xgb, precision_bert2_xgb)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_bert2_xgb, precision_bert2_xgb, label="BERT + XGBoost: {:.3f}".format(auc_precision_recall_bert2_xgb))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for BERT + XGBoost", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+XGBoostPRCurvewithHyp.jpg", dpi = 600)
print("---BERT + XGBoost + PR Curve plot was saved.")

print("----------------------------------------------")

########################################## BERT + 1D-CNN PART ###########################################

x = np.load("HUMAN2EMBEDDINGSX.npy")
y = np.load("HUMAN2EMBEDDINGSY.npy")

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.10, random_state = 42)

import tensorflow as tf


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

a = np.asarray(xtrain).reshape(len(np.asarray(xtrain)),43*768,1)

history1_ = model1.fit(np.asarray(xtrain).reshape(len(np.asarray(xtrain)),43*768,1), utils.to_categorical(ytrain,2),
                    validation_data=(np.asarray(xtest).reshape(len(np.asarray(xtest)),43*768,1), utils.to_categorical(ytest,2)),
                    epochs=25, batch_size=20, verbose=1)    # epoch = 15

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_PR_Curve_Human2 - After_CD-HIT"

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
bert2_probs_cnn1 = cnn_predictions1[:,1]
cnn_predictions1 = np.argmax(cnn_predictions1, axis = 1)
confusion_matrix = confusion_matrix(ytest, cnn_predictions1)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 1D CNN Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+1DCNNConfusionMatrix.jpg", dpi = 600)
print("--- BERT + 1D CNN confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_cnn1_hyp, tpr_keras_bert_cnn1_hyp, _ = roc_curve(ytest, bert2_probs_cnn1)

from sklearn.metrics import auc
auc_keras_bert_cnn1_hyp = auc(fpr_keras_bert_cnn1_hyp, tpr_keras_bert_cnn1_hyp)
print("AUC Score", auc_keras_bert_cnn1_hyp)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_cnn1_hyp, tpr_keras_bert_cnn1_hyp, label="BERT + 1D-CNN: {:.3f}".format(auc_keras_bert_cnn1_hyp))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 2D CNN", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+1DCNNROCCurvewithHyp.jpg", dpi = 600)
print("---BERT + 1D CNN + Hyperparameter Tuning ROC Curve plot was saved.")

print("----------------------------------------------")


precision_bert2_cnn1, recall_bert2_cnn1, _ = precision_recall_curve(ytest, bert2_probs_cnn1)
auc_precision_recall_bert2_cnn1 = auc(recall_bert2_cnn1, precision_bert2_cnn1)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_bert2_cnn1, precision_bert2_cnn1, label="BERT + 1D-CNN: {:.3f}".format(auc_precision_recall_bert2_cnn1))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for BERT + 1D CNN", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+1DCNNPRCurvewithHyp.jpg", dpi = 600)
print("---BERT + 1D CNN + Hyperparameter Tuning PR Curve plot was saved.")

print("----------------------------------------------")



######################################## BERT2OME PART ############################################

x = np.load("HUMAN2EMBEDDINGSX.npy")
y = np.load("HUMAN2EMBEDDINGSY.npy")


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

#importing random search
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

model_2D_ht.summary()

history3 = model_2D_ht.fit(np.asarray(xtest).reshape(len(np.asarray(xtest)),768,43,1), utils.to_categorical(ytest,2),
                           epochs=20, batch_size = 20, validation_split=0.1,
                           initial_epoch=1)     # epochs=15

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_PR_Curve_Human2 - After_CD-HIT"


plt.clf()
plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("Accuracy", fontsize = 15)
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
cnn_predictions3 = model_2D_ht.predict(np.asarray(xtest).reshape(len(np.asarray(xtest)),768,43,1)) # flattened x
bert2_probs = cnn_predictions3[:,1]
print(cnn_predictions3[:,1])    # [prob of 0s, prob of 1s]
cnn_predictions3 = np.argmax(cnn_predictions3, axis = 1)
print(cnn_predictions3)
confusion_matrix = confusion_matrix(ytest, cnn_predictions3)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 2D CNN + H. Tuning Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+2DCNNConfusionMatrixwithHyp.jpg", dpi = 600)
print("--- BERT + 2D CNN + Hyperparameter Tuning confusion matrix plot was saved.")


from sklearn.metrics import roc_curve
print(cnn_predictions3)
fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp, thresholds_keras = roc_curve(ytest, bert2_probs)

from sklearn.metrics import auc
auc_keras_bert_cnn2_hyp = auc(fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp)
print("AUC Score", auc_keras_bert_cnn2_hyp)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp, label="BERT + 2D-CNN: {:.3f}".format(auc_keras_bert_cnn2_hyp))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 2D CNN", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+2DCNNROCCurvewithHyp.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning ROC Curve plot was saved.")

print("----------------------------------------------")


precision_bert2, recall_bert2, _ = precision_recall_curve(ytest, bert2_probs)
auc_precision_recall_bert2 = auc(recall_bert2, precision_bert2)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_bert2, precision_bert2, label="BERT + 2D-CNN: {:.3f}".format(auc_precision_recall_bert2))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for BERT + 2D CNN", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+2DCNNPRCurvewithHyp.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning PR Curve plot was saved.")

print("----------------------------------------------")


###################################### BERT2OME with Chemical Properties PART ########################################

import pandas as pd
import matplotlib.pyplot as plt
import torch
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from tensorflow.keras import utils

import logging

rmbase_dataset_ = pd.read_excel("RMBase_800_CD-HIT.xlsx")
# print(rmbase_dataset_.shape)

y = rmbase_dataset_.label
x = rmbase_dataset_.drop(["label"], axis = 1)

print(x[0][0])

x_sentences = []
for i in range(0, len(x)):
    x_sentences.append("[CLS] " + x[0][i] + " [SEP]")

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

x_sentences_tokenized = []
for i in range(0, len(x_sentences)):
    tokenized_text = tokenizer.tokenize(str(x_sentences[i]))
    x_sentences_tokenized.append(tokenized_text)

print(x_sentences_tokenized[0])

x_sentences_indexes = []
for i in range(0, len(x_sentences)):
    x_sentences_indexes.append(tokenizer.convert_tokens_to_ids(x_sentences_tokenized[i]))

x_segment_ids = []

for i in range(0, int(len(x_sentences)/2)):
    x_segment_ids.append([1] * len(x_sentences_tokenized[0]))
    x_segment_ids.append([0] * len(x_sentences_tokenized[0]))

x_tokens_tensor = torch.tensor([x_sentences_indexes])

x_segments_tensors = torch.tensor([x_segment_ids])

model_x = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model_x.eval()

outputs_x = []

with torch.no_grad():
    outputs_x = model_x(x_tokens_tensor[0], x_segments_tensors[0])
    hidden_states_x = outputs_x[2]

print ("Number of layers:", len(hidden_states_x), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states_x[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states_x[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states_x[layer_i][batch_i][token_i]))

import numpy as np

dataset_x = np.array([])

for j in range(0, len(x)):
    a = np.array([])
    for i in range(0,43):
        b = (np.array(hidden_states_x[12][j][i]) + np.array(hidden_states_x[11][j][i]) + np.array(hidden_states_x[10][j][i]) + np.array(hidden_states_x[9][j][i])) / 4
        a = np.hstack((a,b))
    if len(dataset_x) == 0:
        dataset_x = a
    else:
        dataset_x = np.vstack((dataset_x, a))
dataset_x.shape
dataset_x = pd.DataFrame(dataset_x)

rmbase_dataset_ = pd.read_excel("RMBase_800_CD-HIT.xlsx")
# print(rmbase_dataset_.shape)

y = rmbase_dataset_.label
x = rmbase_dataset_.drop(["label"], axis = 1)

y_values = y
x_values = x

# A -> (1,1,1)
# C -> (0,1,0)
# G -> (1,0,0)
# U -> (0,0,1)

def convert_chemical_property(each_row):
    chemical_vector = []
    chemical_vector = np.array(chemical_vector)

    for i in each_row:
        if i == "A":
            chemical_vector = np.concatenate((chemical_vector, [1, 1, 1]))
        elif i == "C":
            chemical_vector = np.concatenate((chemical_vector, [0, 1, 0]))
        elif i == "G":
            chemical_vector = np.concatenate((chemical_vector, [1, 0, 0]))
        elif i == "U":
            chemical_vector = np.concatenate((chemical_vector, [0, 0, 1]))

    return chemical_vector


all_chemical_vectors = []
for i in range(0, len(x_values)):
    row_vector = convert_chemical_property(x_values[0][i])
    all_chemical_vectors.append(row_vector)
# print(all_chemical_vectors)

chemical_dataframe = pd.DataFrame(all_chemical_vectors)

chemical_dataset_direct_concat = pd.concat([dataset_x, chemical_dataframe], ignore_index=True, axis=1)

dataframe_padding = pd.DataFrame(np.zeros(len(chemical_dataframe)))

dataframe_padding = pd.concat([dataframe_padding, dataframe_padding, dataframe_padding], ignore_index=True, axis=1)

chemical_dataset = pd.concat([dataframe_padding, chemical_dataframe, dataframe_padding], ignore_index=True, axis=1)

big_chemical_dataset = pd.concat([dataset_x.iloc[:, 0:768], chemical_dataset.iloc[:, 0:3]], ignore_index=True, axis=1)
big_chemical_dataset.head()
print(big_chemical_dataset.shape)

jump = 3

for i in range(1, 43):
    big_chemical_dataset = pd.concat(
        [big_chemical_dataset, dataset_x.iloc[:, 768 * i: 768 * (i + 1)], chemical_dataset.iloc[:, jump:jump + 3]],
        ignore_index=True, axis=1)
    jump += 3

# BERT + 2D-CNN Model (Average of Kast 4 Layers) + Chemical Properties + Hyperparameter Tuning

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D

def build_model_ch(hp):
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
            input_shape=(771, 43, 1)),

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
tuner_ch = RandomSearch(build_model_ch,
                        objective="val_accuracy",
                        max_trials = 5, directory = "output", project_name = "AfterHyperParameterTuning_")

xtrain_ch, xtest_ch, ytrain_ch, ytest_ch = train_test_split(big_chemical_dataset, y_values, test_size = 0.10, random_state=42)


tuner_ch.search(np.asarray(xtrain_ch).reshape(len(np.asarray(xtrain_ch)),771,43,1),
              utils.to_categorical(ytrain_ch,2),
              epochs = 8,
              validation_split = 0.2)

model_2D_ht_ch = tuner_ch.get_best_models(num_models=1)[0]

print(model_2D_ht_ch.summary())

history4 = model_2D_ht_ch.fit(np.asarray(xtest_ch).reshape(len(np.asarray(xtest_ch)),771,43,1), utils.to_categorical(ytest_ch,2),
                           epochs=23, batch_size = 20, validation_split=0.1,
                           initial_epoch=1)   # epoch = 15

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_PR_Curve_Human2 - After_CD-HIT"

plt.clf()
plt.plot(history4.history["accuracy"])
plt.plot(history4.history["val_accuracy"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("Accuracy", fontsize = 15)
plt.title("2D CNN Model Accuracy", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNAccuracywithHypwithCh.jpg", dpi = 600)
print("---BERT + 2D CNN + Chemical Properties + Hyperparameter Tuning Accuracy plot was saved.")


plt.clf()
plt.plot(history4.history["loss"])
plt.plot(history4.history["val_loss"])
plt.xlabel("Epoch", fontsize = 15)
plt.ylabel("Loss", fontsize = 15)
plt.title("2D CNN Model Loss", fontsize = 20)
plt.legend(["Train", "Test"], loc="upper left",  prop={"size": 15})
plt.savefig(directory + "/BERT+2DCNNLosswithHypwithCh.jpg", dpi = 600)
print("---BERT + 2D CNN + Chemical Properties + Hyperparameter Tuning Loss plot was saved.")


plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
cnn_predictions4 = model_2D_ht_ch.predict(np.asarray(xtest_ch).reshape(len(np.asarray(xtest_ch)),771,43,1))
bert2_ch_probs = cnn_predictions4[:,1]

cnn_predictions4 = np.argmax(cnn_predictions4, axis = 1)
confusion_matrix4 = confusion_matrix(ytest_ch, cnn_predictions4)
sns.heatmap(confusion_matrix4, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 2D CNN + H. Tuning Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+2DCNNConfusionMatrixwithHypwithCh.jpg", dpi = 600)
print("--- BERT + 2D CNN + Chemical Properties + Hyperparameter Tuning confusion matrix plot was saved.")


print("----------------------------------------------")

from sklearn.metrics import roc_curve
fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch, _ = roc_curve(ytest_ch, bert2_ch_probs)

from sklearn.metrics import auc
auc_keras_bert_cnn2_hyp_ch = auc(fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch)
print("AUC Score", auc_keras_bert_cnn2_hyp_ch)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch, label="BERT + 2D-CNN: {:.3f}".format(auc_keras_bert_cnn2_hyp_ch))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 2D CNN + CH", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+2DCNNROCCurvewithHypCh.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning Ch ROC Curve plot was saved.")

print("----------------------------------------------")


precision_bert2_ch, recall_bert2_ch, _ = precision_recall_curve(ytest_ch, bert2_ch_probs)
auc_precision_recall_bert2_ch = auc(recall_bert2_ch, precision_bert2_ch)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(recall_bert2_ch, precision_bert2_ch, label="BERT + 2D-CNN: {:.3f}".format(auc_precision_recall_bert2_ch))
plt.xlabel("Recall", fontsize = 15)
plt.ylabel("Precision", fontsize = 15)
plt.title("PR curve for BERT + 2D CNN + CH", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+2DCNNPRCurvewithHypCh.jpg", dpi = 600)
print("---BERT + 2D CNN + Hyperparameter Tuning Ch PR Curve plot was saved.")

print("----------------------------------------------")
print("----------------------------------------------")



directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_PR_Curve_Human2 - After_CD-HIT"

# AUC

plt.clf()
plt.plot([0,1],[0,1], 'k--')

plt.plot(fpr_keras_dt, tpr_keras_dt, label="Decision Tree: {:.3f}".format(auc_keras_dt))
plt.plot(fpr_keras_rf, tpr_keras_rf, label="Random Forest: {:.3f}".format(auc_keras_rf))
plt.plot(fpr_keras_xgb, tpr_keras_xgb, label="XGBoost: {:.3f}".format(auc_keras_xgb))

plt.plot(fpr_keras_bert_rf, tpr_keras_bert_rf, label="BERT + Random Forest: {:.3f}".format(auc_keras_bert_rf))
plt.plot(fpr_keras_bert_xgb, tpr_keras_bert_xgb, label="BERT + XGBoost: {:.3f}".format(auc_keras_bert_xgb))
plt.plot(fpr_keras_bert_cnn1_hyp, tpr_keras_bert_cnn1_hyp, label="BERT + 1D-CNN: {:.3f}".format(auc_keras_bert_cnn1_hyp))
plt.plot(fpr_keras_bert_cnn2_hyp, tpr_keras_bert_cnn2_hyp, label="BERT2OMe: {:.3f}".format(auc_keras_bert_cnn2_hyp))
plt.plot(fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch, label="BERT2OMe with Chemical Properties: {:.3f}".format(auc_keras_bert_cnn2_hyp_ch))

plt.legend(loc=4, prop={'size': 8})
plt.xlabel("False Positive Rate", fontsize = 20)
plt.ylabel("True Positive Rate", fontsize = 20)
plt.title("ROC Curve", fontsize = 20)
plt.savefig(directory + "/SumOfROCCurve.jpg", dpi = 600)

print("---Sum of ROC Curve plot was saved.")

# PR

plt.clf()
plt.plot([0,1],[0,1], 'k--')

plt.plot(recall_dt, precision_dt, label="Decision Tree: {:.3f}".format(auc_precision_recall_dt))
plt.plot(recall_rf, precision_rf, label="Random Forest: {:.3f}".format(auc_precision_recall_rf))
plt.plot(recall_xgb, precision_xgb, label="XGBoost: {:.3f}".format(auc_precision_recall_xgb))


plt.plot(recall_bert2_rf, precision_bert2_rf, label="BERT + Random Forest: {:.3f}".format(auc_precision_recall_bert2_rf))
plt.plot(recall_bert2_xgb, precision_bert2_xgb, label="BERT + XGBoost: {:.3f}".format(auc_precision_recall_bert2_xgb))
plt.plot(recall_bert2_cnn1, precision_bert2_cnn1, label="BERT + 1D-CNN: {:.3f}".format(auc_precision_recall_bert2_cnn1))
plt.plot(recall_bert2, precision_bert2, label="BERT2OMe: {:.3f}".format(auc_precision_recall_bert2))
plt.plot(recall_bert2_ch, precision_bert2_ch, label="BERT2OMe with Chemical Properties: {:.3f}".format(auc_precision_recall_bert2_ch))

plt.legend(loc=4, prop={'size': 8})
plt.xlabel("Recall", fontsize = 20)
plt.ylabel("Precision", fontsize = 20)
plt.title("PR Curve", fontsize = 20)
plt.savefig(directory + "/SumOfPRCurve.jpg", dpi = 600)

print("---Sum of PR Curve plot was saved.")



