import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

x = np.load("DNAEMBEDDINGSX.npy")
y = np.load("DNAEMBEDDINGSY.npy")

print("-------- BERT + Random Forest Model --------")

rfc = RandomForestClassifier(random_state=0)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)  #Tree sayısı,  Node ekle ...

rfc.fit(xtrain, ytrain)
bert_rf_predictions = rfc.predict(xtest)

print("Model accuracy score: " + str(accuracy_score(bert_rf_predictions, ytest)))

f1_score = f1_score(ytest, bert_rf_predictions)
print("f1_score:", f1_score)

print("Precision:", metrics.precision_score(bert_rf_predictions, ytest))
print("Recall:", metrics.recall_score(bert_rf_predictions, ytest))

plt.clf()
import seaborn as sns
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, bert_rf_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + Random Forest Confusion Matrix", fontsize = 20)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_DNA"

plt.savefig(directory + "/BERT+RandomForestConfusionMatrix.jpg", dpi = 600)
print("--- BERT + Decision Tree confusion matrix plot was saved.")

from sklearn.metrics import roc_curve
fpr_keras_bert_rf, tpr_keras_bert_rf, thresholds_keras_bert_rf = roc_curve(ytest, bert_rf_predictions)

from sklearn.metrics import auc
auc_keras_bert_rf = auc(fpr_keras_bert_rf, tpr_keras_bert_rf)
print("AUC Score:", auc_keras_bert_rf)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_rf, tpr_keras_bert_rf, label="Random Forest AUC = {:.3f})".format(auc_keras_bert_rf))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + Random Forest", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+RandomForestROCCurve.jpg", dpi = 600)
print("---BERT + Random Forest ROC Curve plot was saved.")

print("----------------------------------------------")

print("-------- BERT + XGBoost Model --------")

from xgboost import XGBClassifier
xgmodel = XGBClassifier()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.10, random_state = 42)

xgmodel.fit(xtrain, ytrain, verbose=False)

bert_xgb_predictions = xgmodel.predict(xtest)

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
fpr_keras_bert_xgb, tpr_keras_bert_xgb, thresholds_keras_bert_xgb = roc_curve(ytest, bert_xgb_predictions)

from sklearn.metrics import auc
auc_keras_bert_xgb = auc(fpr_keras_bert_xgb, tpr_keras_bert_xgb)
print("AUC Score:", auc_keras_bert_xgb)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_xgb, tpr_keras_bert_xgb, label="XGBoost AUC = {:.3f})".format(auc_keras_bert_xgb))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + XGBoost", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+XGBoostROCCurve.jpg", dpi = 600)
print("---BERT + XGBoost ROC Curve plot was saved.")

print("----------------------------------------------")


print("-------- BERT + XGBoost Model + Hyperparameter Tuning --------")

xgmodel_hyp = XGBClassifier()

params = {
 "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
 "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}

random_cv = RandomizedSearchCV(xgmodel_hyp,param_distributions=params,n_iter=5,
                               n_jobs=-1, cv=5,verbose=3,
                               scoring = "roc_auc",
                               return_train_score = True,
                               random_state=42)

random_cv.fit(xtrain, ytrain)

xgmodel_hyp = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3,
              enable_categorical=False, gamma=0.4, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.15, max_delta_step=0, max_depth=15,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, predictor='auto',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)

xgmodel_hyp.fit(xtrain, ytrain, verbose=False)

bert_xgb_hyp_predictions = xgmodel_hyp.predict(xtest)

print("Model accuracy score: " + str(accuracy_score(bert_xgb_hyp_predictions, ytest)))
print("f1 score: ", metrics.f1_score(bert_xgb_hyp_predictions, ytest))
print("Precision:", metrics.precision_score(bert_xgb_hyp_predictions, ytest))
print("Recall:", metrics.recall_score(bert_xgb_hyp_predictions, ytest))

plt.clf()
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(ytest, bert_xgb_hyp_predictions)
sns.heatmap(confusion_matrix, annot = True, fmt = "d", cbar = False)
plt.title("BERT + XGBoost + H.Tuning Confusion Matrix", fontsize = 20)
plt.savefig(directory + "/BERT+XGBoost+HyperparameterTuningConfusionMatrix.jpg", dpi = 600)
print("---BERT + XGBoost + Hyperparameter Tuning confusion matrix plot was saved.")


from sklearn.metrics import roc_curve
fpr_keras_bert_xgb_hyp, tpr_keras_bert_xgb_hyp, thresholds_keras_bert_xgb_hyp = roc_curve(ytest, bert_xgb_hyp_predictions)

from sklearn.metrics import auc
auc_keras_bert_xgb_hyp = auc(fpr_keras_bert_xgb_hyp, tpr_keras_bert_xgb_hyp)
print(auc_keras_bert_xgb_hyp)


plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_xgb_hyp, tpr_keras_bert_xgb_hyp, label="XGBoost AUC = {:.3f})".format(auc_keras_bert_xgb_hyp))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + XGBoost + H.Tuning", fontsize = 20)
plt.legend(loc="best",  prop={"size": 15})
plt.savefig(directory + "/BERT+XGBoost+HyperparameterTuningROCCurve.jpg", dpi = 600)
print("---BERT + XGBoost + Hyperparameter Tuning ROC Curve plot was saved.")

print("----------------------------------------------")