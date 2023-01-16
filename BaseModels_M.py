import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

M = pd.read_excel("M.musculus.xlsx")

y = M.label
x = M.drop(["label"], axis = 1)

y = y.to_list()


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
directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_M"

# BASE MODELS:
print("-------- Decision Tree Model --------")

dt = DecisionTreeClassifier(random_state = 0)

xtrain, xtest, ytrain, ytest = train_test_split(new_x, y, test_size = 0.20, random_state=42)
dt.fit(xtrain, ytrain)
decision_tree_predictions = dt.predict(xtest)

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
fpr_keras_dt, tpr_keras_dt, thresholds_keras_dt = roc_curve(ytest, decision_tree_predictions)

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

print("-------- Random Forest Model --------")

import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

rfc = RandomForestClassifier(random_state=0, criterion="entropy")

rfc.fit(xtrain, ytrain)
random_forest_predictions = rfc.predict(xtest)

print("Model accuracy score: " + str(accuracy_score(random_forest_predictions, ytest)))

f1_score = f1_score(ytest, random_forest_predictions)
print("f1_score:", f1_score)

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
fpr_keras_rf, tpr_keras_rf, thresholds_keras_rf = roc_curve(ytest, random_forest_predictions)

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

print("-------- XGBoost Model --------")

from xgboost import XGBClassifier

xgmodel = XGBClassifier()

xgmodel.fit(xtrain, ytrain, verbose=False)

xgboost_predictions = xgmodel.predict(xtest)

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
fpr_keras_xgb, tpr_keras_xgb, thresholds_keras_xgb = roc_curve(ytest, xgboost_predictions)

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

print("-------- SVM Model --------")

xtrain, xtest, ytrain, ytest = train_test_split(new_x, y, test_size = 0.3, random_state=42)

from sklearn.svm import SVC

clf = SVC(kernel = "linear")
clf.fit(xtrain, ytrain)

svm_predictions = clf.predict(xtest)

print("Model accuracy score: " + str(accuracy_score(svm_predictions, ytest)))
print("f1 score: ", metrics.f1_score(svm_predictions, ytest))
print("Precision:", metrics.precision_score(svm_predictions, ytest))
print("Recall:", metrics.recall_score(svm_predictions, ytest))

plt.clf()
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, svm_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("SVM Confusion Matrix", fontsize = 20)
plt.savefig(directory + "/SVMConfusionMatrix.jpg", dpi = 600)
print("---SVM confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_svm, tpr_keras_svm, thresholds_keras_svm = roc_curve(ytest, svm_predictions)

from sklearn.metrics import auc
auc_keras_svm = auc(fpr_keras_svm, tpr_keras_svm)
print("AUC Score:", auc_keras_svm)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_svm, tpr_keras_svm, label='AUC (area = {:.3f})'.format(auc_keras_svm))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for SVM", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/SVMROCCurve.jpg", dpi = 600)
print("---SVM ROC Curve plot was saved.")

print("----------------------------------------------")

plt.clf()
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr_keras_dt, tpr_keras_dt, label = "Decision Tree: %0.6f" % auc_keras_dt)
plt.plot(fpr_keras_rf, tpr_keras_rf, label = "Random Forest: %0.6f" % auc_keras_rf)
plt.plot(fpr_keras_xgb, tpr_keras_xgb, label = "XGBoost: %0.6f" % auc_keras_xgb)
plt.plot(fpr_keras_svm, tpr_keras_svm, label = "SVM: %0.6f" % auc_keras_svm)
plt.legend(loc=4, prop={'size': 13})
plt.xlabel("False Positive Rate", fontsize = 15)
plt.ylabel("True Positive Rate", fontsize = 15)
plt.title("ROC Curve", fontsize = 20)
plt.savefig(directory + "/BaseModelsROCCurve.jpg", dpi = 600)

print("---Base Models ROC Curve plot was saved.")







