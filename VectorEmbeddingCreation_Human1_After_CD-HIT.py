import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch


test_neg = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetCDHIT\test_new_neg\H_N", header = None)
test_pos = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetCDHIT\test_new_pos\H_P", header = None)
train_neg = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetCDHIT\train_new_neg\H_N", header = None)
train_pos = pd.read_csv(r"C:\Users\nisas\OneDrive\Masaüstü\TezDatasetCDHIT\train_new_pos\H_P", header = None)

x = pd.concat([train_pos, train_neg, test_pos, test_neg], ignore_index=True)
print(x)
print(x.shape)
print(type(x))

train_pos_labels = [1] * len(train_pos)
train_neg_labels = [0] * len(train_neg)
test_pos_labels = [1] * len(test_pos)
test_neg_labels = [0] * len(test_neg)

# Defining labels
y = train_pos_labels + train_neg_labels + test_pos_labels + test_neg_labels

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

print("Data type of x: ",type(x))
print("Data type of y: ", type(y))

directory_x = r"C:\Users\nisas\PycharmProjects\RNA-DNA-Nisa\HUMAN1EMBEDDINGSX.npy"
np.save(directory_x, dataset_x)

directory_y = r"C:\Users\nisas\PycharmProjects\RNA-DNA-Nisa\HUMAN1EMBEDDINGSY.npy"
np.save(directory_y, y)








