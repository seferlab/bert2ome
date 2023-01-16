import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

test_neg = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetsAfterModification\test_new_neg\H_N", header = None)
test_pos = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetsAfterModification\test_new_pos\H_P", header = None)
train_neg = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetsAfterModification\train_new_neg\H_N", header = None)
train_pos = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetsAfterModification\train_new_pos\H_P", header = None)

import torch
from transformers import BertTokenizer, BertModel

# In order to see more information on what's happening we activate the logger as follows:
import logging

# NEGATIVE TEST DATASET

test_neg_sentences = []
for i in range(0, len(test_neg)):
    test_neg_sentences.append("[CLS] " + test_neg[0][i] + " [SEP]")

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

test_neg_sentences_tokenized = []
for i in range(0, len(test_neg_sentences)):
    tokenized_text = tokenizer.tokenize(str(test_neg_sentences[i]))
    test_neg_sentences_tokenized.append(tokenized_text)

test_neg_sentences_indexes = []
for i in range(0, len(test_neg_sentences)):
    test_neg_sentences_indexes.append(tokenizer.convert_tokens_to_ids(test_neg_sentences_tokenized[i]))

test_neg_segment_ids = []

for i in range(0, int(len(test_neg_sentences)/2)):
    test_neg_segment_ids.append([1] * len(test_neg_sentences_tokenized[0]))
    test_neg_segment_ids.append([0] * len(test_neg_sentences_tokenized[0]))

test_neg_tokens_tensor = torch.tensor([test_neg_sentences_indexes])

test_neg_segments_tensors = torch.tensor([test_neg_segment_ids])

model_test_neg = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model_test_neg.eval()
print("Test neg. evaluated.")

outputs_test_neg = []

with torch.no_grad():
    outputs_test_neg = model_test_neg(test_neg_tokens_tensor[0], test_neg_segments_tensors[0])
    hidden_states_test_neg = outputs_test_neg[2]

print ("Number of layers:", len(hidden_states_test_neg), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states_test_neg[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states_test_neg[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states_test_neg[layer_i][batch_i][token_i]))

print("VECTOR EMBEDDINGS FOR NEGATIVE TEST DATASET WERE CREATED..")

# POSITIVE TEST DATASET

test_pos_sentences = []
for i in range(0, len(test_pos)):
    test_pos_sentences.append("[CLS] " + test_pos[0][i] + " [SEP]")

test_pos_sentences_tokenized = []
for i in range(0, len(test_pos_sentences)):
    tokenized_text = tokenizer.tokenize(str(test_pos_sentences[i]))
    test_pos_sentences_tokenized.append(tokenized_text)

test_pos_sentences_indexes = []
for i in range(0, len(test_pos_sentences)):
    test_pos_sentences_indexes.append(tokenizer.convert_tokens_to_ids(test_pos_sentences_tokenized[i]))

test_pos_segment_ids = []

for i in range(0, int(len(test_pos_sentences)/2)):
    test_pos_segment_ids.append([1] * len(test_pos_sentences_tokenized[0]))
    test_pos_segment_ids.append([0] * len(test_pos_sentences_tokenized[0]))

test_pos_tokens_tensor = torch.tensor([test_pos_sentences_indexes])

test_pos_segments_tensors = torch.tensor([test_pos_segment_ids])

model_test_pos = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model_test_pos.eval()

outputs_test_pos = []

with torch.no_grad():
    outputs_test_pos = model_test_pos(test_pos_tokens_tensor[0], test_pos_segments_tensors[0])
    hidden_states_test_pos = outputs_test_pos[2]

print ("Number of layers:", len(hidden_states_test_pos), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states_test_pos[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states_test_pos[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states_test_pos[layer_i][batch_i][token_i]))

print("VECTOR EMBEDDINGS FOR POSITIVE TEST DATASET WERE CREATED..")

# NEGATIVE TRAIN DATASET

train_neg_sentences = []
for i in range(0, len(train_neg)):
    train_neg_sentences.append("[CLS] " + train_neg[0][i] + " [SEP]")

train_neg_sentences_tokenized = []
for i in range(0, len(train_neg_sentences)):
    tokenized_text = tokenizer.tokenize(str(train_neg_sentences[i]))
    train_neg_sentences_tokenized.append(tokenized_text)

train_neg_sentences_indexes = []
for i in range(0, len(train_neg_sentences)):
    train_neg_sentences_indexes.append(tokenizer.convert_tokens_to_ids(train_neg_sentences_tokenized[i]))

train_neg_segment_ids = []

for i in range(0, int(len(train_neg_sentences)/2)):
    train_neg_segment_ids.append([1] * len(train_neg_sentences_tokenized[0]))
    train_neg_segment_ids.append([0] * len(train_neg_sentences_tokenized[0]))
train_neg_segment_ids.append([1] * len(train_neg_sentences_tokenized[0]))

train_neg_tokens_tensor = torch.tensor([train_neg_sentences_indexes])

train_neg_segments_tensors = torch.tensor([train_neg_segment_ids])

model_train_neg = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model_train_neg.eval()

outputs_train_neg = []

with torch.no_grad():
    outputs_train_neg = model_train_neg(train_neg_tokens_tensor[0], train_neg_segments_tensors[0])
    hidden_states_train_neg = outputs_train_neg[2]

outputs_train_neg

print ("Number of layers:", len(hidden_states_train_neg), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states_train_neg[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states_train_neg[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states_train_neg[layer_i][batch_i][token_i]))

print("VECTOR EMBEDDINGS FOR NEGATIVE TRAIN DATASET WERE CREATED..")

# POSITIVE TRAIN DATASET

train_pos_sentences = []
for i in range(0, len(train_pos)):
    train_pos_sentences.append("[CLS] " + train_pos[0][i] + " [SEP]")

train_pos_sentences_tokenized = []
for i in range(0, len(train_pos_sentences)):
    tokenized_text = tokenizer.tokenize(str(train_pos_sentences[i]))
    train_pos_sentences_tokenized.append(tokenized_text)

train_pos_sentences_indexes = []
for i in range(0, len(train_pos_sentences)):
    train_pos_sentences_indexes.append(tokenizer.convert_tokens_to_ids(train_pos_sentences_tokenized[i]))

train_pos_segment_ids = []

for i in range(0, int(len(train_pos_sentences)/2)):
    train_pos_segment_ids.append([1] * len(train_pos_sentences_tokenized[0]))
    train_pos_segment_ids.append([0] * len(train_pos_sentences_tokenized[0]))
train_pos_segment_ids.append([1] * len(train_pos_sentences_tokenized[0]))

train_pos_tokens_tensor = torch.tensor([train_pos_sentences_indexes])

train_pos_segments_tensors = torch.tensor([train_pos_segment_ids])

model_train_pos = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model_train_pos.eval()

outputs_train_pos = []

with torch.no_grad():
    outputs_train_pos = model_train_pos(train_pos_tokens_tensor[0], train_pos_segments_tensors[0])
    hidden_states_train_pos = outputs_train_pos[2]


print ("Number of layers:", len(hidden_states_train_pos), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states_train_pos[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states_train_pos[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states_train_pos[layer_i][batch_i][token_i]))

print("VECTOR EMBEDDINGS FOR POSITIVE TRAIN DATASET WERE CREATED..")

import numpy as np

dataset_train_pos = np.array([])

for j in range(0, len(train_pos)):
    a = np.array([])
    for i in range(0,43):
        b = (np.array(hidden_states_train_pos[12][j][i]) + np.array(hidden_states_train_pos[11][j][i]) + np.array(hidden_states_train_pos[10][j][i]) + np.array(hidden_states_train_pos[9][j][i])) / 4
        a = np.hstack((a,b))
    if len(dataset_train_pos) == 0:
        dataset_train_pos = a
    else:
        dataset_train_pos = np.vstack((dataset_train_pos, a))
dataset_train_pos.shape
dataset_train_pos = pd.DataFrame(dataset_train_pos)


dataset_train_neg = np.array([])

for j in range(0, len(train_neg)):
    a = np.array([])
    for i in range(0,43):
        b = (np.array(hidden_states_train_neg[12][j][i]) + np.array(hidden_states_train_neg[11][j][i]) + np.array(hidden_states_train_neg[10][j][i]) + np.array(hidden_states_train_neg[9][j][i])) / 4
        a = np.hstack((a,b))
    if len(dataset_train_neg) == 0:
        dataset_train_neg = a
    else:
        dataset_train_neg = np.vstack((dataset_train_neg, a))
dataset_train_neg.shape
dataset_train_neg = pd.DataFrame(dataset_train_neg)


dataset_test_pos = np.array([])

for j in range(0, len(test_pos)):
    a = np.array([])
    for i in range(0,43):
        b = (np.array(hidden_states_test_pos[12][j][i]) + np.array(hidden_states_test_pos[11][j][i]) + np.array(hidden_states_test_pos[10][j][i]) + np.array(hidden_states_test_pos[9][j][i])) / 4
        a = np.hstack((a,b))
    if len(dataset_test_pos) == 0:
        dataset_test_pos = a
    else:
        dataset_test_pos = np.vstack((dataset_test_pos, a))
dataset_test_pos.shape
dataset_test_pos = pd.DataFrame(dataset_test_pos)


dataset_test_neg = np.array([])

for j in range(0, len(test_neg)):
    a = np.array([])
    for i in range(0,43):
        b = (np.array(hidden_states_test_neg[12][j][i]) + np.array(hidden_states_test_neg[11][j][i]) + np.array(hidden_states_test_neg[10][j][i]) + np.array(hidden_states_test_neg[9][j][i])) / 4
        a = np.hstack((a,b))
    if len(dataset_test_neg) == 0:
        dataset_test_neg = a
    else:
        dataset_test_neg = np.vstack((dataset_test_neg, a))
dataset_test_neg.shape
dataset_test_neg = pd.DataFrame(dataset_test_neg)

print("Dataset train positive:", dataset_train_pos.shape)
print("Dataset train negative:", dataset_train_neg.shape)
print("Dataset test positive:", dataset_test_pos.shape)
print("Dataset test negative:", dataset_test_neg.shape)

dataset_train_pos_labels = [1] * len(train_pos)
dataset_train_neg_labels = [0] * len(train_neg)
dataset_test_pos_labels = [1] * len(test_pos)
dataset_test_neg_labels = [0] * len(test_neg)

x = pd.concat([dataset_train_pos, dataset_train_neg, dataset_test_pos, dataset_test_neg], ignore_index=True)
y = dataset_train_pos_labels + dataset_train_neg_labels + dataset_test_pos_labels + dataset_test_neg_labels

print("Data type of x: ",type(x))
print("Data type of y: ", type(y))

directory_x = r"C:\Users\nisas\PycharmProjects\RNA-DNA-Nisa\RNAEMBEDDINGSX.npy"
np.save(directory_x, x)

directory_y = r"C:\Users\nisas\PycharmProjects\RNA-DNA-Nisa\RNAEMBEDDINGSY.npy"
np.save(directory_y, y)








