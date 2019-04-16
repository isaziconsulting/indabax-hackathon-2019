# encoding: utf-8

'''
Encoding, decoding, string and data utils.
'''

import torch
from glob import glob
from os import path
import numpy as np
import json
import csv


def decode_output(output, labels):
    '''Converts from softmax model output into a letter/sequence of letters.

    Args:decode_words(output, labels)
        output (tensor): Prediction from model as a pytorch tensor.
        labels (list): List of labels for model.

    Returns:
        string: Output from model as a string.
    '''
    np_output = output.permute(1,2,0).detach().numpy()
    # (N, F, t)
    W = np_output.shape[2]
    # use max of softmax output vector as the chosen class per column
    idx = np.argmax(np_output, axis=1)
    words = []
    for batch_idx in range(np_output.shape[0]):
        # collect groups of tokens
        groups = []
        for i in range(W):
            if i == 0 or idx[batch_idx][i] != idx[batch_idx][i - 1]:
                groups += [idx[batch_idx][i]-1]
        # translate numeric tokens to char equivalent
        word = ''
        for i in range(len(groups)):
            if groups[i] != -1:
                word += labels[groups[i]]
        words.append(word)
    return words

def decode_label_words(words, labels):
    '''Converts sequences of numeric labels to letters.

    Args:
        words (list of tensors): List of encoded words.
        labels (list): List of labels for model.

    Returns:
        string: Output from model as a string.
    '''
    decoded = []
    for word in words:
        decoded.append("".join([labels[l-1] for l in word]))
    return decoded

def encode_words(words, labels):
    '''Converts sequences of letters to numeric labels for ctc loss.

    Args:
        words (list of strings): Words.
        labels (list): List of labels for model.

    Returns:
        string: Output from model as a string.
    '''
    encoded_words = []
    for word in words:
        encoded_word = [labels.index(l)+1 for l in word]
        encoded_words.append(torch.tensor(encoded_word, dtype=torch.long))
    return encoded_words

def norm_levenshtein_dist(str1, str2):
    '''Finds the levenshtein edit distance between two strings normalised to the len of strings being compared
    Based on https://rosettacode.org/wiki/Levenshtein_distance#Python

    Args:
        str1 (string): 1st string
        str2 (string): 2nd string
    
    Returns:
        float: Normalised edit distance
    '''
    m = len(str1)
    n = len(str2)
    d = [[i] for i in range(m+1)]
    d[0] = [i for i in range(n+1)]
    for j in range(1, n+1):
        for i in range(1, m+1):
            if str1[i-1] == str2[j-1]:
                d[i].insert(j,d[i-1][j-1])
            else:
                minimum = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+2)
                d[i].insert(j, minimum)
    ldist = d[-1][-1]
    norm = ldist/float(m + n)
    return norm

def save_model(model, path):
    '''save your model in a desired path 

    Args:
        path (string): path name
    '''
    torch.save(model.state_dict(), path)

def save_predictions(path, data_list, preds):
    '''save predictions as csv for submission

    Args:
        path (string): path name
        data_list (string): path for data being predicted
        preds (list of strings): data predictions
    '''
    with open(data_list) as f:
        names = json.load(f)
    with open(path, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(list(zip(names, preds)))