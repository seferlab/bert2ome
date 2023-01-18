import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import torch
from transformers import BertTokenizer, BertModel

# In order to see more information on what's happening we activate the logger as follows:
import logging

# rmbase_dataset_ = pd.read_excel("RMBase_Big.xlsx")
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




directory_x = r"C:\Users\nisas\PycharmProjects\RNA-DNA-Nisa\HUMAN2EMBEDDINGSX.npy"
np.save(directory_x, dataset_x)

directory_y = r"C:\Users\nisas\PycharmProjects\RNA-DNA-Nisa\HUMAN2EMBEDDINGSY.npy"
np.save(directory_y, y)