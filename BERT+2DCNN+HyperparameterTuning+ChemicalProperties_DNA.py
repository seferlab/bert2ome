import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import torch
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

from tensorflow.keras import utils

# In order to see more information on what's happening we activate the logger as follows:
import logging

# rmbase_dataset_ = pd.read_excel("RMBase_Big.xlsx")
rmbase_dataset_ = pd.read_excel("RMBase_800.xlsx")
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

rmbase_dataset_ch = pd.read_excel("RMBase_800.xlsx")
print(rmbase_dataset_ch.head(5))

y_values = rmbase_dataset_ch.label
x_values = rmbase_dataset_ch.drop(["label"], axis = 1)

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
#    print(768 * i)
#    print(68 * (i + 1))
#    print(big_chemical_dataset.shape)

# BERT + 2D-CNN Model (Average of Kast 4 Layers) + Chemical Properties + Hyperparameter Tuning

#importing tensorflow
import tensorflow as tf
#importing keras from tensorflow
from tensorflow import keras
# importing Sequential from keras
from tensorflow.keras.models import Sequential
#importing Dense and Conv2D layers from keras
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
                           epochs=15, batch_size = 20, validation_split=0.1,
                           initial_epoch=1)

directory = r"C:\Users\nisas\OneDrive\Masaüstü\Thesis_Results_Nisa_DNA"

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
cnn_predictions4 = np.argmax(cnn_predictions4, axis = 1)
confusion_matrix4 = confusion_matrix(ytest_ch, cnn_predictions4)
sns.heatmap(confusion_matrix4, annot = True, fmt = "d", cbar = False)
plt.title("BERT + 2D CNN + H. Tuning Confusion Matrix", fontsize = 20)

plt.savefig(directory + "/BERT+2DCNNConfusionMatrixwithHypwithCh.jpg", dpi = 600)
print("--- BERT + 2D CNN + Chemical Properties + Hyperparameter Tuning confusion matrix plot was saved.")


from sklearn.metrics import roc_curve
fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch, thresholds_keras_ch = roc_curve(ytest_ch, cnn_predictions4)

from sklearn.metrics import auc
auc_keras_bert_cnn2_hyp_ch = auc(fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch)
print("AUC Score", auc_keras_bert_cnn2_hyp_ch)

plt.clf()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_bert_cnn2_hyp_ch, tpr_keras_bert_cnn2_hyp_ch, label='Keras (area = {:.3f})'.format(auc_keras_bert_cnn2_hyp_ch))
plt.xlabel("False positive rate", fontsize = 15)
plt.ylabel("True positive rate", fontsize = 15)
plt.title("ROC curve for BERT + 2D CNN", fontsize = 20)
plt.legend(loc="best",  prop={'size': 15})
plt.savefig(directory + "/BERT+2DCNNROCCurvewithHypwithCh.jpg", dpi = 600)
print("---BERT + 2D CNN + Chemical Properties + Hyperparameter Tuning ROC Curve plot was saved.")

print("----------------------------------------------")

