import sys

import math
import os

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from tensorflow.keras import *
# from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.python.ops.gen_math_ops import Sigmoid
from tqdm import tqdm

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.layers import Layer

from tensorflow.keras.optimizers import Adam, SGD

import csv
import numpy as np
import tensorflow.keras.utils as kutils

from tensorflow import keras, sigmoid

def read_csv():
    data = pd.read_csv("./Human dataset/PELM_S_data.csv")
    print(data)


def getMatrixLabel(positive_position_file_name, sites, window_size=49, empty_aa='*'):
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = [] # list of raw protein sequence
    all_label = [] # list of phos site label

    short_seqs = [] # list of short sequence
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        num = 0
        for row in reader:
            # print("row[0]:",int(row[0]))
            # if num == 1128:
            #     print('1')

            # if num == 226:
            #     print(1)

            position = int(row[2])
            sseq = row[3]
            rawseq.append(row[3])
            center = sseq[position - 1]
            if center in sites:
                all_label.append(int(row[0]))
                # print("length of all_label",len(all_label))
                prot.append(row[1])
                pos.append(row[2])

                if position - half_len > 0:
                    start = position - half_len
                    start = int(start)
                    position = int(position)
                    left_seq = sseq[start - 1:position - 1]
                else:

                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = position + half_len
                    end = int(end)
                right_seq = sseq[position:end]

                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    nb_lack = int(nb_lack)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    nb_lack = int(nb_lack)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
                shortseq = left_seq + center + right_seq
                short_seqs.append(shortseq)
                # num = num + 1
                # print(num)
        targetY = kutils.to_categorical(all_label)
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict["*"] = 20
        # letterDict["?"] = 21
        Matr = np.zeros((len(short_seqs), window_size))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            # print(samplenumber)
            for AA in seq:
                if AA not in letterDict:
                    # AANo += 1
                    continue
                Matr[samplenumber][AANo] = letterDict[AA]
                AANo = AANo + 1
            samplenumber = samplenumber + 1
    # print('data process finished')
    print("matr.shape", Matr.shape)
    return Matr, targetY, all_label

def get_peptide(sequence, win=21):
    start = int((len(sequence) - 1) / 2 - (win - 1) / 2)
    end = int((len(sequence) - 1) / 2 + (win - 1) / 2 + 1)
    peptide = sequence[start:end]
    return peptide

blosum62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, 0],  # A
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, 0],  # R
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 0],  # N
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 0],  # D
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, 0],  # C
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0],  # Q
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 0],  # E
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, 0],  # G
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0],  # H
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, 0],  # I
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, 0],  # L
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0],  # K
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, 0],  # M
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, 0],  # F
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, 0],  # P
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0],  # S
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, 0],  # T
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, 0],  # W
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, 0],  # Y
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, 0],  # V
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
}
# 从AAindex数据库中选取了8个理化特性分别是
# Hydrophobicity index, Mean polarity, Isoelectric point, Refractivity, Average flexibility indices
# Average volume of buried residue, Transfer free energy to surface, Consensus normalized hydrophobility scale
AAindex = {
    'A': [0.61, -0.06, 6.00,  4.34,  0.357, 91.50, -0.20, 0.25],  # A
    'Q': [0.00, -0.73, 10.76, 26.66, 0.529, 202.0, -0.12, -1.76],  # Q
    'R': [0.60, -0.84, 5.41,  13.28, 0.463, 135.2, -0.08, -0.64],  # R
    'N': [0.06, -0.48, 2.77,  12.00, 0.511, 124.5, -0.20, -0.72],  # N
    'D': [0.46, -0.80, 5.05,  35.77, 0.346, 117.7, -0.45, 0.04],  # D
    'C': [1.07, 1.36,  5.65,  17.56, 0.493, 161.1, 0.16,  -0.69],  # C
    'E': [0.47, -0.77, 3.22,  17.26, 0.497, 155.1, -0.30, -0.62],  # E
    'G': [0.07, -0.41, 5.97,  0.00,  0.544, 66.40, 0.00,  0.16],  # G
    'H': [0.61, 0.49,  7.59,  21.81, 0.323, 167.3, -0.12, -0.40],  # H
    'I': [2.22, 1.31,  6.02,  19.06, 0.462, 168.8, -2.26, 0.73],  # I
    'L': [1.53, 1.21,  5.98,  18.78, 0.365, 167.9, -2.46, 0.53],  # L
    'K': [1.15, -1.18, 9.74,  21.29, 0.466, 171.3, -0.35, -1.10],  # K
    'M': [1.18, 1.27,  5.74,  21.64, 0.295, 170.8, -1.47, 0.26],  # M
    'F': [2.02, 1.27,  5.48,  29.40, 0.314, 203.4, -2.33, 0.61],  # F
    'P': [1.95, 0.00,  6.30,  10.93, 0.509, 129.3, -0.98, -0.07],  # P
    'S': [0.05, -0.50, 5.68,  6.35,  0.507, 99.10, -0.39, -0.26],  # S
    'T': [0.05, -0.27, 5.66,  11.01, 0.444, 122.1, -0.52, -0.18],  # T
    'W': [2.65, 0.88,  5.89,  42.53, 0.305, 237.6, -2.01, 0.37],  # W
    'Y': [1.88, 0.33,  5.66,  31.53, 0.420, 203.6, -2.24, 0.02],  # Y
    'V': [2.32, 1.09,  5.96,  13.92, 0.386, 141.7, -1.56, 0.54],  # V
    'X': [0.00, 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],  # X
}
# Blosum62进化特征
def blosum62_matrix(peptide):
    blosum62_encode = []
    peptide = peptide.upper()
    for x in peptide:
        if blosum62.get(x) == None:
            # print(x)
            blosum62_encode.append(blosum62.get('X'))
        else:
            blosum62_encode.append(blosum62.get(x))
    return blosum62_encode
def get_blosum62(filename,win=21):
    df = pd.read_csv(filename, sep=',', header=None)
    df.loc[:,4]= np.nan
    df[4] = df[4].astype(object)
    # df.info()
    for i in tqdm(range(len(df))):
        sequence = df.iloc[i,3]
        index = df.iloc[i,2]
        #
        if (index <= (win - 1) / 2 and index >= len(sequence) - (win - 1) / 2):
            sequence = int((win - 1) / 2) * 'X' + sequence +int((win - 1) / 2) * 'X'
            index = index + int((win - 1) / 2)
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i, 4] = blosum62_matrix(peptide)
        elif index <= (win - 1) / 2:
            sequence = int((win - 1) / 2) * 'X' + sequence
            index = index + int((win - 1) / 2)
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i,4] = blosum62_matrix(peptide)
        elif (index >= len(sequence) - (win - 1) / 2):
            sequence = sequence + int((win - 1) / 2) * 'X'
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i, 4] = blosum62_matrix(peptide)
        else:
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i, 4] = blosum62_matrix(peptide)

    # df.to_csv('./S_'+ str(win) +'_blosum62.csv',header=False,index=False)
    Blosum62 = df.iloc[:, 4]
    # AAindex = np.array(AAindex)
    Blosum62 = [np.array(i) for i in Blosum62]
    # for x in Blosum62:
    #     if x.shape != (21,21):
    #         print(x)
    Blosum62 = np.array(Blosum62)
    return Blosum62

# AAindex理化特征
def AAindex_matrix(peptide,AAindex):
    peptide = peptide.upper()
    AAindex_encode = []
    for x in peptide:
        if AAindex.get(x) == None:
            AAindex_encode.append(AAindex.get('X'))
        else:
            AAindex_encode.append(AAindex.get(x))
    return AAindex_encode
def get_AAindex(filename,AAindex,win=21):
    # df = pd.read_csv('./Human dataset/aaindex_feature.txt', sep='\s+')
    # # 去除含有空值的列
    # df = df.dropna(axis=1, how='any')
    # # df.to_csv('./AAindex_feature.csv', header=False, index=False)
    # AAindex = {'X': 531 * [0]}
    # for i in range(len(df)):
    #     residue = df.iloc[i, 0]
    #     feature = df.iloc[i, 1:].values.tolist()
    #     AAindex[residue] = feature
    df = pd.read_csv(filename, sep=',', header=None)
    df.loc[:,4]= np.nan
    df[4] = df[4].astype(object)
    # df.info()
    for i in tqdm(range(len(df))):
        sequence = df.iloc[i,3]
        index = df.iloc[i,2]
        if (index <= (win - 1) / 2 and index >= len(sequence) - (win - 1) / 2):
            sequence = int((win - 1) / 2) * 'X' + sequence +int((win - 1) / 2) * 'X'
            index = index + int((win - 1) / 2)
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i, 4] = AAindex_matrix(peptide,AAindex)
        if index <= (win - 1) / 2:
            sequence = int((win - 1) / 2) * 'X' + sequence
            index = index + int((win - 1) / 2)
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i,4] = AAindex_matrix(peptide,AAindex)
        elif (index >= len(sequence) - (win - 1) / 2):
            sequence = sequence + int((win - 1) / 2) * 'X'
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i, 4] = AAindex_matrix(peptide,AAindex)
        else:
            peptide = sequence[index - 1 - int((win - 1) / 2):index + int((win - 1) / 2)]
            df.at[i, 4] = AAindex_matrix(peptide,AAindex)
    # AAindex_feature = df.loc[1,4]
    # AAindex_feature.to
    # df.to_csv('./S_' + str(win) + '_AAindex.csv',header=False,index=False)
    AAindex = df.iloc[:,4]
    # AAindex = np.array(AAindex)
    AAindex =np.array([np.array(i) for i in AAindex])
    return AAindex


def getMatrixLabel_batch(batch_set, sites, window_size=49, empty_aa='*'):
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = [] # list of raw protein sequence
    all_label = [] # list of phos site label

    short_seqs = [] # list of short sequence
    half_len = (window_size - 1) / 2

    for index, row in batch_set.iterrows():

        position = int(row[3].split("-")[1])
        sseq = row[4]
        rawseq.append(row[3])
        center = sseq[position - 1]
        if center in sites:
            all_label.append(int(row[0]))
            # print("length of all_label",len(all_label))
            prot.append(row[1])
            pos.append(row[3])

            if position - half_len > 0:
                start = position - half_len
                start = int(start)
                position = int(position)
                left_seq = sseq[start - 1:position - 1]
            else:

                left_seq = sseq[0:position - 1]

            end = len(sseq)
            if position + half_len < end:
                end = position + half_len
                end = int(end)
            right_seq = sseq[position:end]

            if len(left_seq) < half_len:
                nb_lack = half_len - len(left_seq)
                nb_lack = int(nb_lack)
                left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

            if len(right_seq) < half_len:
                nb_lack = half_len - len(right_seq)
                nb_lack = int(nb_lack)
                right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
            shortseq = left_seq + center + right_seq
            short_seqs.append(shortseq)

    targetY = kutils.to_categorical(all_label)
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["*"] = 20
    # letterDict["?"] = 21
    Matr = np.zeros((len(short_seqs), window_size))
    samplenumber = 0
    for seq in short_seqs:
        AANo = 0
        # print(samplenumber)
        for AA in seq:
            if AA not in letterDict:
                # AANo += 1
                continue
            Matr[samplenumber][AANo] = letterDict[AA]
            AANo = AANo + 1
        samplenumber = samplenumber + 1
    # print('data process finished')
    # print("matr.shape", Matr.shape)
    return Matr, targetY, all_label

def get_batch(data_df,batch_size_half):
    batch_num = len(data_df) // batch_size_half
    batches = np.array_split(data_df.iloc[:batch_num*batch_size_half],batch_num)
    if len(data_df) % batch_size_half != 0 :
        batch_add = data_df.iloc[:batch_num*batch_size_half].sample(n=batch_size_half-len(data_df) % batch_size_half )
        batch = pd.concat([data_df.iloc[batch_num*batch_size_half:],batch_add],axis=0,ignore_index=True)
    batches.append(batch)
    return batches
# 构建平衡的batchset集合
def get_batch_list(positive,negative,batch_size_half):
    # positive_batch_num = math.ceil(len(positive) / batch_size_half)
    # negative_batch_num = math.ceil(len(negative) / batch_size_half)
    positive_batch = get_batch(positive, batch_size_half)
    negative_batch = get_batch(negative, batch_size_half)
    batch_set_list = []
    # for pos in tqdm(positive_batch, total=(len(positive_batch))):
    for i in tqdm(range(len(negative_batch))):
        # j = i%len(positive_batch)
        batch_set = pd.concat([positive_batch[i%len(positive_batch)], negative_batch[i]]).sample(frac=1)
        batch_set_list.append(batch_set)

    # batch_list = []
    #
    # for i in tqdm(range(int(positive_batch_num))):
    #     if ((i + 1) * batch_size_half <= len(positive)):
    #         batch_positive = positive.iloc[i * batch_size_half:(i + 1) * batch_size_half, :]
    #     else:
    #         batch_positive = positive.iloc[i * batch_size_half:, :]
    #         # 从positive中随机采样补充到batchsize大小
    #         batch_positive_add = positive.iloc[:i * batch_size_half, :].sample(n=batch_size_half - len(batch_positive))
    #         batch_positive = pd.concat([batch_positive_add, batch_positive], axis=0, ignore_index=True)
    #
    #     for j in range(int(negative_batch_num)):
    #         if ((j + 1) * batch_size_half <= len(negative)):
    #             batch_negative = negative.iloc[j * batch_size_half:(j + 1) * batch_size_half, :]
    #         else:
    #             batch_negative = negative.iloc[j * batch_size_half:, :]
    #             batch_negative_add = negative.iloc[:j * batch_size_half, :].sample(n=batch_size_half - len(batch_negative))
    #             batch_negative = pd.concat([batch_negative_add, batch_negative], axis=0, ignore_index=True)
    #
    #         batch_set = pd.concat([batch_negative, batch_positive]).sample(frac=1)
    #         batch_list.append(batch_set)
    return batch_set_list

def getmatrix(x, w, L):
    matrix = []
    n = []
    for j in range(w):
        n = tf.reshape(x[:, :, j * w:(j + 1) * w], (-1, L * w))
        print(n.shape)
        matrix.append(n)
    matrix = tf.concat(matrix, axis=1)
    matrix = tf.reshape(matrix, (-1, L * w, w))
    return matrix

def plot_history(history,save_dictory,residue):
    save_path = './train_img/' + save_dictory + '/' + residue + '/'
    isExists = os.path.exists(save_path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(save_path)
        print("successfully")
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print("existed")
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    hist = history

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train loss')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val loss')
    plt.ylim([0.2, 1])
    plt.legend()

    # plt.savefig(save_path + 'loss.jpg')
    plt.savefig(save_path + 'loss2.jpg')
    # plt.savefig(save_path + 'loss3.jpg')

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Train accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label='Val accuracy')
    plt.ylim([0.3, 1])
    plt.legend()
    # plt.savefig(save_path + 'accuracy.jpg')
    plt.savefig(save_path + 'accuracy2.jpg')
    # plt.savefig(save_path + 'accuracy3.jpg')

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('auc')
    plt.plot(hist['epoch'], hist['auc'],
             label='Train auc')
    plt.plot(hist['epoch'], hist['val_auc'],
             label='Val auc')
    plt.ylim([0.3, 1])
    plt.legend()
    # plt.savefig(save_path + 'auc.jpg')
    plt.savefig(save_path + 'auc2.jpg')
    # plt.savefig(save_path + 'auc3.jpg')
    plt.show()


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='concat', **kwargs):

        self.size = size

        self.mode = mode

        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x,**kwargs):

        if (self.size == None) or (self.mode == 'concat'):
            self.size = int(x.shape[-1])
        # print(x.shape)
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)

        position_j = K.expand_dims(position_j, 0)
        # print(position_j.shape)

        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1

        position_i = K.expand_dims(position_i, 2)
        # print(position_i.shape)
        position_ij = K.dot(position_i, position_j)
        #         print(K.cos(position_ij).shape, K.sin(position_ij).shape)

        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        #         print(position_ij.shape)

        if self.mode == 'sum':

            return position_ij + x

        elif self.mode == 'concat':

            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):

        if self.mode == 'sum':

            return input_shape

        elif self.mode == 'concat':

            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size,
            'mode': self.mode,
        })
        return config


class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)

    def call(self, x, **kwargs):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (self.output_dim ** 0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
        })
        return config

class SE_Net(Layer):
    def __init__(self,units,ration,**kwargs):
        super(SE_Net,self).__init__(**kwargs)
        self.global_avg_pool = GlobalAveragePooling1D()
        self.conv1 = Conv1D(filters=4,kernel_size=1,strides=1,padding='valid',activation='relu')
        self.conv2 = Conv1D(filters=16, kernel_size=1, strides=1, padding='valid', activation='sigmoid')
        # Excitation 操作：两个全连接层
        self.units = units
        self.ration = ration
        self.fc1 = Dense(units // ration,
                         activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-5))
        self.fc2 = Dense(units,
                         activation='sigmoid',
                         kernel_initializer='he_normal',
                         bias_initializer='zeros',
                         kernel_regularizer=l2(1e-5))
    def call(self, inputs, *args, **kwargs):
        squeeze_1 = self.global_avg_pool(inputs)
        # squeeze_1 = Lambda(squeeze_1)

        excitation_1 = self.fc1(squeeze_1)
        excitation_1 = self.fc2(excitation_1)
        # x_scale = tf.expand_dims(excitation_1, axis=1)
        # x_scale = tf.expand_dims(x_scale, axis=1)
        x_weighted = tf.multiply(inputs,tf.expand_dims(excitation_1,axis=1))

        return x_weighted
    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(SE_Net, self).get_config()
        config.update({
            'units':self.units,
            'ration':self.ration
        })
        return config

class ECA(Layer):
    def __init__(self, channel, gamma=2, b=1,**kwargs):  #64
        super(ECA, self).__init__(**kwargs)
        self.channel = channel
        self.gamma = gamma
        self.b = b
        kernel_size = int(abs((math.log(channel,2)+  b)/gamma))  #3
        kernel_size = kernel_size if kernel_size % 2  else kernel_size+1  #3
        padding = kernel_size//2
        self.avg_pool = GlobalAveragePooling1D(keepdims=True)
        self.conv =  Conv1D(1, kernel_size=kernel_size,padding='same',name="eca_conv")
        self.sigmoid = Activation('sigmoid', name='eca_conv1_relu_')

    def call(self, x, **kwargs):
        # b, l, w = x.size()
        # print("ECA\n")
        # print(x.shape) #(None, 3, 64)
        #变成序列的形式
        avg = self.avg_pool(x)#(None, 1, 64)
        avg = Reshape((self.channel, 1))(avg) #(None, 64, 1)
        # print(avg.shape)
        weight = self.conv(avg)#(None, 64, 1)
        # print(weight.shape)
        weight = self.sigmoid(weight)#(None, 64, 1)
        weight = Reshape((1,self.channel))(weight)#(None, 1, 64)
        # print(weight.shape)

        x_weight = tf.multiply(x,weight)#(None, 3, 64)
        # print(x_weight)
        return  x_weight
    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(ECA, self).get_config()
        config.update({
            'channel':self.channel,
            'gamma':self.gamma,
            'b':self.b
        })
        return config

class Res_Block(Layer):
    def __init__(self, num_filters, kernel_size, dilation_rate, dropout_rate=0.0, **kwargs):
        super(Res_Block, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.SE_net = SE_Net(units=64)
        self.ECA = ECA(64)
        # 验证输入是否为可迭代对象
        if not isinstance(self.num_filters, (list, tuple)):
            raise ValueError("num_filters must be a list or tuple")
        if not isinstance(self.kernel_size, (list, tuple)):
            raise ValueError("kernel_size must be a list or tuple")
        if not isinstance(self.dilation_rate, (list, tuple)):
            raise ValueError("dilation_rate must be a list or tuple")

        self.conv1 = Conv1D(filters=num_filters[0],
                           kernel_size=kernel_size[0],
                           dilation_rate=dilation_rate[0],
                           padding='causal',
                           use_bias=False)
        self.conv2 = Conv1D(filters=num_filters[1],
                           kernel_size=kernel_size[1],
                           dilation_rate=dilation_rate[1],
                           padding='causal',
                           use_bias=False)
        self.conv3 = Conv1D(filters=num_filters[2],
                           kernel_size=kernel_size[2],
                           dilation_rate=dilation_rate[2],
                           padding='causal',
                           use_bias=False)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.relu = ReLU()
        self.res_conv = Conv1D(filters=num_filters[2],kernel_size=1,padding='same')
        if dropout_rate > 0.0:
            self.dropout = Dropout(dropout_rate)
        else:
            self.dropout = None

    def call(self, inputs, training=None,**kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        x = self.conv3(x)
        x = self.ECA(x)
        x = self.bn3(x, training=training)
        x = self.relu(x)
        # if self.dropout is not None:
        #     x = self.dropout(x, training=training)
        x_res = self.res_conv(inputs)
        x = Add()([x,x_res])
        x = self.relu(x)
        return x
    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(Res_Block, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config
class TCN(Layer):
    def __init__(self,num_filters,kernel_size,dilation_rate,dropout_rate,filter_num,**kwargs):
        super(TCN,self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.filter_num = filter_num
        self.ResBlock = Res_Block(num_filters, kernel_size, dilation_rate, dropout_rate=dropout_rate)
        self.conv1d = Conv1D(kernel_size=1,filters=filter_num)

    def call(self, inputs, *args, **kwargs):
        # inputs shape:(batch_size,win,16)
        win = inputs.shape[1]
        forward_input = tf.reshape(inputs,(-1,win*4,4))
        # forward_input = inputs
        # forward
        # forward_outputs = [self.ResBlock(forward_input),self.ResBlock(forward_input),self.ResBlock(forward_input)]
        # forward_output = tf.reduce_mean(tf.stack(forward_outputs,axis=0),axis=0)
        forward_output = self.ResBlock(forward_input)
        # print(forward_outputs.shape,forward_outputs.dtype)
        # backward  注意这个concat 是在 filter_number 维度进行 即 输出的形状是类似于 (batch_size,sequence,filter_number)
        # 三个维度的顺序与实际情况可能有差别，最后进行调整
        backward_inputs = tf.reverse(inputs,axis=[1])
        backward_inputs = tf.reshape(backward_inputs,(-1,win*4,4))
        # backward_outputs = [self.ResBlock(backward_inputs),self.ResBlock(backward_inputs),self.ResBlock(backward_inputs)]
        # backward_output = tf.reduce_mean(tf.stack(backward_outputs,axis=0),axis=0)
        backward_output = self.ResBlock(backward_inputs)
        # print(backward_outputs.shape,backward_outputs.dtype)
        # 融合前向和后向信息
        outputs = tf.concat([forward_output, backward_output], axis=-1)
        # print(outputs.shape,outputs.dtype)
        outputs = self.conv1d(outputs)

        return outputs

    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(TCN, self).get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'filter_num': self.filter_num
        })
        return config



# from transformer_layer import Encoder
from transformer_layer import Encoder
from Capsule_net import CapsNet, margin_loss

def getmodel(win1, win2, win3, max_features,num_filters,kernel_size,dilation_rate,filter_num,dropout_rate=0.3,self_number=16,init_form = 'RandomUniform',weight_decay = 0.0001):

    # Encoderlayer = transformer_encoder(num_layers=2, num_heads=4, dff=2048, d_model=16, input_vocab_size=max_features)
    inputs1 = Input(shape=(win1,), dtype='int32')
    # x1 = Encoderlayer(inputs1) # (None,21,16)
    x1 = Embedding(max_features, 16)(inputs1)
    #
    # x1 = Position_Embedding(mode='sum')(x1)
    # print(x1.shape)
    # # self_number = 16
    # # x1 = Self_Attention(self_number)(x1)
    # x1 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x1,x1)
    # x1=getmatrix(x1,4,win1)
    # x1 = tf.reshape(x1, (-1, win1 * 4, 4))
    # print(x1.shape)



    inputs2 = Input(shape=(win2,), dtype='int32')
    # x2 = Encoderlayer(inputs2) # (None,33,16)
    x2 = Embedding(max_features, 16)(inputs2)
    #
    # x2 = Position_Embedding(mode='sum')(x2)
    # # self_number = 16
    # # x2 = Self_Attention(self_number)(x2)
    # x2 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x2,x2)

    # x2=getmatrix(x2,4,win2)
    # x2 = tf.reshape(x2, (-1, win2 * 4, 4))
    # print(x2.shape)



    inputs3 = Input(shape=(win3,), dtype='int32')
    # x3 = Encoderlayer(inputs3) # (None,51,16)
    x3 = Embedding(max_features, 16)(inputs3)
    #
    # x3 = Position_Embedding(mode='sum')(x3)
    # print(x1.shape)
    # # self_number = 16
    # # x3 = Self_Attention(self_number)(x3)
    # x3 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x3,x3)

    # x3=getmatrix(x3,4,win3)
    # x3 = tf.reshape(x3, (-1, win3 * 4, 4))

    # print(x3.shape)



    # init_form = 'RandomUniform'
    # weight_decay = 0.0001
    print(x1.shape)

    x1 = Conv1D(64, 12, kernel_initializer=init_form, strides=4, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x1) # (None,3,64)
    x1 = ECA(64)(x1)

    x2 = Conv1D(64, 12, kernel_initializer=init_form, strides=4, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x2) # (None,6,64)
    x2 = ECA(64)(x2)
    x3 = Conv1D(64, 12, kernel_initializer=init_form, strides=4, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x3) # (None,10,64)
    x3 = ECA(64)(x3)

    # TCN
    # x1 = TCN(num_filters,kernel_size,dilation_rate,dropout_rate,filter_num)(x1)
    # x2 = TCN(num_filters,kernel_size,dilation_rate,dropout_rate,filter_num)(x2)
    # x3 = TCN(num_filters,kernel_size,dilation_rate,dropout_rate,filter_num)(x3)

    x = keras.layers.concatenate((x1, x2, x3), axis=1) # (None,19,64)

    x = CapsNet(n_class=2,routings=3)(x)
    print(x)

    # x1 = Flatten()(x)
    # x = Dense(256, activation='sigmoid')(x1)
    # if dropout_rate > 0:
    #     x = Dropout(dropout_rate)(x)
    # x = Dense(84, activation='sigmoid')(x)
    # if dropout_rate > 0:
    #     x = Dropout(dropout_rate)(x)
    # x = Dense(2, activation='softmax')(x)
    x = Activation('softmax')(x)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=x, name="res-net")
    print(model.summary())
    # learn_rate_scheduler = keras.optimizers.schedules.LearningRateSchedule()
    opt = Adam(learning_rate=1e-4)
    # opt = Adam(learning_rate=0.000001, decay=0.00001)
    loss = 'categorical_crossentropy'
    # loss = margin_loss
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy', 'AUC', 'Precision'],
                  )
    return model

import tensorflow as tf

def euclidean_distance(x, y):
  """计算两个向量的欧氏距离。

  参数：
    x: 第一个向量，shape 为 [batch_size, sequence_length, num_dimensions]。
    y: 第二个向量，shape 为 [batch_size, sequence_length, num_dimensions]。

  返回：
    一个包含欧氏距离的张量，shape 为 [batch_size,num_dimensions]。
  """

  # 计算平方差
  square_diff = tf.square(x - y)

  # 在 sequence_length 维度上求和
  sum_square_diff = tf.reduce_sum(square_diff, axis=1)

  # 计算平方根得到欧氏距离
  distance = tf.sqrt(sum_square_diff)

  return distance
def get_attenphos(win1, win2, win3, max_features,num_filters,kernel_size,dilation_rate,filter_num,dropout_rate=0.3,self_number=16,init_form = 'RandomUniform',learning_rate=1e-5,weight_decay = 0.0001):


    inputs1 = Input(shape=(win1,), dtype='int32')

    x1 = Embedding(max_features, 16)(inputs1)

    x1 = Position_Embedding(mode='sum')(x1)
    # print(x1.shape)
    self_number = 16
    print(self_number/4,type(self_number/4))
    x1 = Self_Attention(self_number)(x1)
    # x1 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x1,x1)
    # x1=getmatrix(x1,4,win1)
    x1 = tf.reshape(x1, (-1, win1 * 4, int(self_number/4)))
    # print(x1.shape)

    inputs2 = Input(shape=(win2,), dtype='int32')
    x2 = Embedding(max_features, 16)(inputs2)
    #
    x2 = Position_Embedding(mode='sum')(x2)
    # self_number = 16
    x2 = Self_Attention(self_number)(x2)
    # x2 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x2,x2)

    # x2=getmatrix(x2,4,win2)
    x2 = tf.reshape(x2, (-1, win2 * 4, int(self_number/4)))
    # print(x2.shape)

    inputs3 = Input(shape=(win3,), dtype='int32')
    x3 = Embedding(max_features, 16)(inputs3)
    #
    x3 = Position_Embedding(mode='sum')(x3)
    # print(x1.shape)
    # self_number = 16
    x3 = Self_Attention(self_number)(x3)
    # x3 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x3,x3)

    # x3=getmatrix(x3,4,win3)
    x3 = tf.reshape(x3, (-1, win3 * 4, int(self_number/4)))

    # print(x3.shape)

    # init_form = 'RandomUniform'
    # weight_decay = 0.0001
    print(x1.shape)

    filter = 64
    kernel_size = 12
    stride = 2
    x1 = Conv1D(filter, kernel_size, kernel_initializer=init_form, strides=stride, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x1)  # (None,3,64)

    # x1 = Flatten()(x1)
    # x1 = Dense(units=256,activation='relu')(x1)

    x2 = Conv1D(filter, kernel_size, kernel_initializer=init_form, strides=stride, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x2)  # (None,6,64)
    # x2 = Flatten()(x2)
    # x2 = Dense(units=256, activation='relu')(x2)

    x3 = Conv1D(filter, kernel_size, kernel_initializer=init_form, strides=stride, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x3)  # (None,10,64)
    # x3 = Flatten()(x3)
    # x3 = Dense(units=256, activation='relu')(x3)


    # print(x1.shape,euclidean_distance_1.shape)
    # x1 = tf.concat([x1,euclidean_distance_1],axis=1)
    # x2 = tf.concat([x2,euclidean_distance_2],axis=1)
    # x3 = tf.concat([x3,euclidean_distance_3],axis=1)
    # print(x1.shape)
    x = keras.layers.concatenate((x1, x2, x3), axis=1)  # (None,19,64)


    x1 = Flatten()(x)
    # x = Dense(256, activation='relu')(x1)
    x = Dense(256, activation='sigmoid')(x1)
    x = Dense(84, activation='sigmoid')(x)
    x = Dense(2, activation='softmax')(x)
    # x = Activation('softmax')(x)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=x, name="res-net")
    # model = Model(inputs=[inputs1, inputs2, inputs3,
    #                       AAindex1,AAindex2,AAindex3,
    #                       Blosum62_1,Blosum62_2,Blosum62_3,
    #                       base_AAindex1,base_AAindex2,base_AAindex3,
    #                       base_blosum62_1,base_blosum62_2,base_blosum62_3], outputs=x, name="res-net")
    print(model.summary())
    # learn_rate_scheduler = keras.optimizers.schedules.LearningRateSchedule()
    opt = Adam(learning_rate=learning_rate,decay=weight_decay)
    # opt = Adam(learning_rate=0.000001, decay=0.00001)
    loss = 'categorical_crossentropy'
    # loss = focal_loss
    # loss = margin_loss
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy', 'AUC', 'Recall'],
                  )
    return model

# use additional feature  such as aaindex and blosum62
def get_attenphos_aaindex(win1, win2, win3, max_features,num_filters,kernel_size,dilation_rate,filter_num,dropout_rate=0.3,self_number=16,init_form = 'RandomUniform',weight_decay = 0.0001):

    aaindex_dim = 8
    blosum62_dim = 21

    activate = 'relu'

    AAindex1 = Input(shape=(win1,aaindex_dim))
    AAindex2 = Input(shape=(win2,aaindex_dim))
    AAindex3 = Input(shape=(win3,aaindex_dim))
    # print('aaindex',AAindex1.shape)

    Blosum62_1 = Input(shape=(win1,blosum62_dim))
    Blosum62_2 = Input(shape=(win2,blosum62_dim))
    Blosum62_3 = Input(shape=(win3,blosum62_dim))
    # print('blosum62',Blosum62_1.shape)

    # combine feature dimension is (batch_size,sequence_length,531(aaindex)+21(blosum62))
    combine_feature1 = tf.concat([AAindex1,Blosum62_1],axis=2)
    combine_feature2 = tf.concat([AAindex2,Blosum62_2],axis=2)
    combine_feature3 = tf.concat([AAindex3,Blosum62_3],axis=2)

    # combine_feature1 = Dense(64)(combine_feature1)
    # combine_feature2 = Dense(64)(combine_feature2)
    # combine_feature3 = Dense(64)(combine_feature3)
    print(combine_feature1.shape)

    base_AAindex1 = Input(shape=(win1,aaindex_dim))
    base_AAindex2 = Input(shape=(win2,aaindex_dim))
    base_AAindex3 = Input(shape=(win3,aaindex_dim))

    base_blosum62_1 = Input(shape=(win1,blosum62_dim))
    base_blosum62_2 = Input(shape=(win2,blosum62_dim))
    base_blosum62_3 = Input(shape=(win3,blosum62_dim))

    base_combine_feature1 = tf.concat([base_AAindex1, base_blosum62_1], axis=-1)
    base_combine_feature2 = tf.concat([base_AAindex2, base_blosum62_2], axis=-1)
    base_combine_feature3 = tf.concat([base_AAindex3, base_blosum62_3], axis=-1)


    euclidean_distance_1 = euclidean_distance(combine_feature1,base_combine_feature1)
    euclidean_distance_2 = euclidean_distance(combine_feature2,base_combine_feature2)
    euclidean_distance_3 = euclidean_distance(combine_feature3,base_combine_feature3)

    print(euclidean_distance_1.shape)
    print(euclidean_distance_2.shape)
    print(euclidean_distance_3.shape)

    euclidean_distance_1 = Dense(units=512,activation=activate)(euclidean_distance_1)
    # euclidean_distance_1 = Dropout(0.5)(euclidean_distance_1)
    euclidean_distance_1 = Dense(units=256)(euclidean_distance_1)
    euclidean_distance_2 = Dense(units=512,activation=activate)(euclidean_distance_2)
    # euclidean_distance_2 = Dropout(0.5)(euclidean_distance_2)
    euclidean_distance_2 = Dense(units=256)(euclidean_distance_2)
    euclidean_distance_3 = Dense(units=512,activation=activate)(euclidean_distance_3)
    # euclidean_distance_3 = Dropout(0.5)(euclidean_distance_3)
    euclidean_distance_3 = Dense(units=256)(euclidean_distance_3)

    print('euclidean_distance',euclidean_distance_1.shape)

    inputs1 = Input(shape=(win1,), dtype='int32')

    x1 = Embedding(max_features, 16)(inputs1)

    x1 = Position_Embedding(mode='sum')(x1)
    # print(x1.shape)
    self_number = 16
    print(self_number/4,type(self_number/4))
    x1 = Self_Attention(self_number)(x1)
    # x1 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x1,x1)
    # x1=getmatrix(x1,4,win1)
    x1 = tf.reshape(x1, (-1, win1 * 4, int(self_number/4)))
    # print(x1.shape)

    inputs2 = Input(shape=(win2,), dtype='int32')
    x2 = Embedding(max_features, 16)(inputs2)
    #
    x2 = Position_Embedding(mode='sum')(x2)
    # self_number = 16
    x2 = Self_Attention(self_number)(x2)
    # x2 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x2,x2)

    # x2=getmatrix(x2,4,win2)
    x2 = tf.reshape(x2, (-1, win2 * 4, int(self_number/4)))
    # print(x2.shape)

    inputs3 = Input(shape=(win3,), dtype='int32')
    x3 = Embedding(max_features, 16)(inputs3)
    #
    x3 = Position_Embedding(mode='sum')(x3)
    # print(x1.shape)
    # self_number = 16
    x3 = Self_Attention(self_number)(x3)
    # x3 = MultiHeadAttention(num_heads=4,value_dim=16,key_dim=16)(x3,x3)

    # x3=getmatrix(x3,4,win3)
    x3 = tf.reshape(x3, (-1, win3 * 4, int(self_number/4)))

    # print(x3.shape)

    # init_form = 'RandomUniform'
    # weight_decay = 0.0001
    print(x1.shape)

    filter = 64
    kernel_size = 12
    stride = 4
    x1 = Conv1D(filter, kernel_size, kernel_initializer=init_form, strides=stride, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x1)  # (None,3,64)

    x1 = Flatten()(x1)
    x1 = Dense(units=256,activation='relu')(x1)

    x2 = Conv1D(filter, kernel_size, kernel_initializer=init_form, strides=stride, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x2)  # (None,6,64)
    x2 = Flatten()(x2)
    x2 = Dense(units=256, activation='relu')(x2)

    x3 = Conv1D(filter, kernel_size, kernel_initializer=init_form, strides=stride, activation='relu',
                padding='valid',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x3)  # (None,10,64)
    x3 = Flatten()(x3)
    x3 = Dense(units=256, activation='relu')(x3)


    # print(x1.shape,euclidean_distance_1.shape)
    x1 = tf.concat([x1,euclidean_distance_1],axis=1)
    x2 = tf.concat([x2,euclidean_distance_2],axis=1)
    x3 = tf.concat([x3,euclidean_distance_3],axis=1)

    # MutualCrossAttention
    # x1 = MutualCrossAttention(0.5,64)(x1,combine_feature1)
    # x2 = MutualCrossAttention(0.5,64)(x2,combine_feature2)
    # x3 = MutualCrossAttention(0.5,64)(x3,combine_feature3)
    print(x1.shape)
    x = keras.layers.concatenate((x1, x2, x3), axis=1)  # (None,19,64)


    x1 = Flatten()(x)
    # x = Dense(256, activation='relu')(x1)
    x = Dense(256, activation='relu')(x1)
    x = Dense(84, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    # x = Activation('softmax')(x)
    # model = Model(inputs=[inputs1, inputs2, inputs3], outputs=x, name="res-net")
    model = Model(inputs=[inputs1, inputs2, inputs3,
                          AAindex1,AAindex2,AAindex3,
                          Blosum62_1,Blosum62_2,Blosum62_3,
                          base_AAindex1,base_AAindex2,base_AAindex3,
                          base_blosum62_1,base_blosum62_2,base_blosum62_3], outputs=x, name="res-net")
    print(model.summary())
    # learn_rate_scheduler = keras.optimizers.schedules.LearningRateSchedule()
    opt = Adam(learning_rate=1e-3,decay=weight_decay)
    # opt = Adam(learning_rate=0.000001, decay=0.00001)
    loss = 'categorical_crossentropy'
    # loss = focal_loss
    # loss = margin_loss
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy', 'AUC', 'Recall'],
                  )
    return model


def gelu_(X):
    return 0.5 * X * (1.0 + tf.math.tanh(0.7978845608028654 * (X + 0.044715 * tf.pow(X, 3))))


class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU

    Y = GELU()(X)

    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.

    Usage: use it as a tf.keras.Layer


    '''

    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


from HybridCBAM import CFEM_Block, TAM,CFEM_Block_noAEM


# TIM loss
# margin loss
def get_entry(probs):
    avg_probs = K.mean(probs,axis=0,keepdims=True)
    ent = -K.sum(avg_probs * (K.log(avg_probs + K.epsilon()) / K.log(K.constant(2))),axis=1,keepdims=True)
    return ent

# condition loss
def get_cond_entropy(probs):
    cond_ent = -K.sum(probs * K.log(probs + K.epsilon()),axis=1)
    cond_ent = K.mean(cond_ent,keepdims=True)
    return cond_ent

def TIM_loss(logits,labels,alpha=1.0):
    loss = losses.categorical_crossentropy(labels,logits)
    loss = K.mean(loss)
    loss = K.abs(loss-alpha) + alpha
    # probs = softmax(logits)
    probs = logits
    sum_loss = loss + get_entry(probs) - get_cond_entropy(probs)
    return sum_loss


def get_CBAMnet(win1,win2,win3,kernel_size,units,units2,max_features=21,pool_size=2,stride=1,reduce=4,drop_rate=0.5,drop_rate1=0.2):
    '''
    :parma win1: first window sequence length
    :parma win2: first window sequence length
    :parma win3: first window sequence length
    :param kernel_size: CFEM Conv1d kernel_size
    :param units: CFEM 1,2,4 conv1d filter_number
    :param units2: CFEM 3 conv1d filter_number
    :param pool_size: CFEM maxpooling  pool_size default: 2
    :param stride:  CFEM maxpooling stride default: 1
    :param reduce: CBAM CAM channel reduce rate default: 4
    :param drop_rate: CFEM 1 Dropout layer droprate default: 0.5
    :param drop_rate1: CFEM 2,3,4 Dropout layer droprate default: 0.2
    :return: model
    '''
    activation = 'tanh'
    outdim = 16

    input = Input(shape=(win1,max_features))
    # x = Embedding(max_features, outdim)(input)
    x = input
    # CFEM 1
    x = CFEM_Block(units=units,reduce=reduce,kernel_size=kernel_size,pool_size=pool_size,stride=stride,drop_rate=drop_rate)(x)

    # CFEM 2
    x = CFEM_Block(units=units,reduce=reduce,kernel_size=kernel_size,pool_size=pool_size,stride=stride,drop_rate=drop_rate1)(x)
    #
    # CFEM 3
    x = CFEM_Block(units=units2,reduce=reduce,kernel_size=kernel_size,pool_size=pool_size,stride=stride,drop_rate=drop_rate1)(x)

    # CFEM 4
    x = CFEM_Block_noAEM(units=units,reduce=reduce,kernel_size=kernel_size,pool_size=pool_size,stride=stride,drop_rate=drop_rate1)(x)

    x = BatchNormalization()(x)
    # BiGRU
    x_gru = Bidirectional(GRU(units=units//2,activation=activation,return_sequences=True))(x)
    x_gru = Bidirectional(GRU(units=units//2,activation=activation,return_sequences=True))(x_gru)

    # TAM
    x_tam = TAM(units=units)(x_gru)

    # BiLSTM
    x_lstm = Bidirectional(LSTM(units=units//2, activation=activation,return_sequences=True))(x)
    x_lstm = Bidirectional(LSTM(units=units//2, activation=activation,return_sequences=True))(x_lstm)
    #
    # # TAM
    x_tam2 = TAM(units=units)(x_lstm)

    # concat
    # out = x_tam
    out = concatenate((x_tam, x_tam2))


    # 33长度的蛋白质序列
    input2 = Input(shape=(win2,max_features))
    # x2 = Embedding(max_features, outdim)(input2)
    # x2_embed = x2
    x2 = input2
    # CFEM 1
    x2 = CFEM_Block(units=units, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
                   drop_rate=drop_rate)(x2)

    # CFEM 2
    x2 = CFEM_Block(units=units, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
                   drop_rate=drop_rate1)(x2)

    # CFEM 3
    x2 = CFEM_Block(units=units2, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
                   drop_rate=drop_rate1)(x2)

    # CFEM 4
    x2 = CFEM_Block_noAEM(units=units, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
                   drop_rate=drop_rate1)(x2)


    x2 = BatchNormalization()(x2)
    # BiGRU
    x2_gru = Bidirectional(GRU(units=units//2, activation=activation, return_sequences=True))(x2)
    x2_gru = Bidirectional(GRU(units=units//2, activation=activation, return_sequences=True))(x2_gru)

    # TAM
    x2_tam = TAM(units=units)(x2_gru)

    # BiLSTM
    x2_lstm = Bidirectional(LSTM(units=units//2, activation=activation, return_sequences=True))(x2)
    x2_lstm = Bidirectional(LSTM(units=units//2, activation=activation, return_sequences=True))(x2_lstm)
    #
    # # TAM
    x2_tam2 = TAM(units=units)(x2_lstm)

    # concat
    # out2 = x2_gru
    out2 = concatenate((x2_tam, x2_tam2))

    # 51长度的蛋白质
    input3 = Input(shape=(win3,max_features))
    # x3 = Embedding(max_features, outdim)(input3)
    # x3_embed = x3
    x3 = input3
    # CFEM 1
    x3 = CFEM_Block(units=units, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
                    drop_rate=drop_rate)(x3)

    # CFEM 2
    # x3 = CFEM_Block(units=units, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
    #                 drop_rate=drop_rate1)(x3)
    #
    # # CFEM 3
    # x3 = CFEM_Block(units=units2, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
    #                 drop_rate=drop_rate1)(x3)
    #
    # # CFEM 4
    # x3 = CFEM_Block_noAEM(units=units, reduce=reduce, kernel_size=kernel_size, pool_size=pool_size, stride=stride,
    #                 drop_rate=drop_rate1)(x3)
    #
    # x3 = BatchNormalization()(x3)
    # # BiGRU
    # x3_gru = Bidirectional(GRU(units=units//2, activation=activation, return_sequences=True))(x3)
    # x3_gru = Bidirectional(GRU(units=units//2, activation=activation, return_sequences=True))(x3_gru)
    #
    # # TAM
    # x3_tam = TAM(units=units)(x3_gru)
    #
    # # BiLSTM
    # x3_lstm = Bidirectional(LSTM(units=units//2, activation=activation, return_sequences=True))(x3)
    # x3_lstm = Bidirectional(LSTM(units=units//2, activation=activation, return_sequences=True))(x3_lstm)
    #
    # # TAM
    # x3_tam2 = TAM(units=units)(x3_lstm)
    #
    # # concat
    # # out3 = x3_gru
    # out3 = concatenate((x3_tam, x3_tam2))


    # 三个不同长度的蛋白质特征融合
    out_concat = keras.layers.concatenate((out, out2), axis=1)
    # out_concat = keras.layers.concatenate((out, out2, out3), axis=1)
    # out_concat = keras.layers.concatenate((x, x2, x3), axis=1)
    # output = CapsNet(n_class=2, routings=3)(out_concat)
    # output = Flatten()(out_concat)
    output = GlobalAveragePooling1D()(out_concat)
    output = Dense(256,activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(84,activation='relu')(output)
    output = Dropout(0.5)(output)
    output = Dense(2,activation='softmax')(output)

    # RSFPN
    # unit = 128
    # act = GELU()
    # output1 = Dense(unit/2,activation=act)(output)
    # output2 = Dense(unit/4,activation=act)(output1)
    # output3 = Dense(unit/8,activation=act)(output2)
    # output4 = Dense(unit/16,activation=act)(output3)
    # output5 = Dense(unit/32,activation=act)(output4)
    # output6 = Dense(unit/16,activation=act)(output5)
    # output7 = Dense(unit/8,activation=act)(output6+output4)
    # output8 = Dense(unit/4,activation=act)(output3+output7)
    # output = Dense(2,activation='softmax')(output8+output2)

    # model = Model(inputs=input, outputs=out, name="HybridCBAM")
    model = Model(inputs=[input,input2], outputs=output, name="HybridCBAM")
    # model = Model(inputs=[input,input2,input3], outputs=output, name="HybridCBAM")
    # model = Model(inputs=[input, input2, input3,
    #                       AAindex1, AAindex2, AAindex3,
    #                       Blosum62_1, Blosum62_2, Blosum62_3,
    #                       base_AAindex1, base_AAindex2, base_AAindex3,
    #                       base_blosum62_1, base_blosum62_2, base_blosum62_3], outputs=output, name="HybridCBAM_aaindex_blosum62")

    # model.build(input_shape=((None, 51), (None, 33), (None, 21)))
    # model.summary()
    print(model.summary())
    # learn_rate_scheduler = keras.optimizers.schedules.LearningRateSchedule()
    opt = Adam(learning_rate=1e-3,decay=0.00001)
    # opt = Adam(learning_rate=0.000001, decay=0.00001)
    loss = 'categorical_crossentropy'
    # loss = 'mean_squared_error'
    # loss = TIM_loss
    # loss = focal_loss
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy', 'AUC', 'Recall'],
                  )
    return model


def cal_metric(y_pred,y_true,residue,mode):
    np.set_printoptions(threshold=sys.maxsize)
    dataset = 'Dataset'
    # save_path = './test_result/' + dataset  + mode + residue[0] + "/"
    save_path = './test_result/' + dataset  + mode + residue + "/"
    isExists = os.path.exists(save_path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(save_path)
        print("successfully")
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print("dictory is existed")
    y_pred_label = np.argmax(y_pred,axis=1)
    y_true_label = np.argmax(y_true,axis=1)
    tn, fp, fn, tp= confusion_matrix(y_true_label,y_pred_label).ravel()
    acc = sklearn.metrics.accuracy_score(y_true_label,y_pred_label)
    f1 = sklearn.metrics.f1_score(y_true_label,y_pred_label)
    mcc = sklearn.metrics.matthews_corrcoef(y_true_label,y_pred_label)
    pre = sklearn.metrics.precision_score(y_true_label,y_pred_label)
    sn = tp/(tp+fn)
    sp = tn/(tn+fp)
    # pre = tp/(tp+fp)
    # f1 = (2*pre*sn)/(pre+sn)
    # mcc = (tp*tn-fp*fn)/math.sqrt((tp+fn)*(tp+fp)*(tn+fn)*(tn+fp))
    y_score = y_pred[:,1]
    auc = roc_auc_score(y_true_label,y_score)

    with open(save_path + 'result_best_2.txt','w') as f:
    # with open(save_path + 'result_2.txt','w') as f:
        f.write(f"CM:{confusion_matrix(y_true_label,y_pred_label)}\n"
                f"sn:{sn}\n"
                f"sp:{sp}\n"
                f"f1:{f1}\n"
                f"mcc:{mcc}\n"
                f"auc:{auc}\n")
        f.write(f"predict_label:{str(y_pred_label)}\n")
        f.write(f"label:{str(y_true_label)}\n")
        f.write(f"predict_prob:{str(y_pred)}\n")
        f.close()
    # roc曲线绘制
    fpr, tpr, thresholds = roc_curve(y_true_label, y_score, pos_label=1)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(save_path + 'roc_curve_best_2.jpg')
    # plt.savefig(save_path + 'roc_curve_2.jpg')
    plt.show()

    # PR曲线绘制
    precision, recall, thresholds = precision_recall_curve(y_true_label,y_score,pos_label=1)
    ap = average_precision_score(y_true_label, y_score)
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % ap)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path + 'pr_curve_best_2.jpg')
    # plt.savefig(save_path + 'roc_curve_2.jpg')
    plt.show()


# Focal_loss
def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)


class MutualCrossAttention(Layer):

    def __init__(self, dropout_rate,hidden_dim,**kwargs):
        super(MutualCrossAttention, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.dropout = Dropout(dropout_rate)
        self.dense = Dense(hidden_dim)

    def call(self, x1, x2, **kwargs):
        # Assign x1 and x2 to query and key
        query = self.dense(x1)
        key = self.dense(x2)
        d = query.shape[-1]

        # Basic attention mechanism formula to get intermediate output A
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(d, tf.float32))
        output_A = tf.matmul(self.dropout(tf.nn.softmax(scores, axis=-1)), x2)

        # Basic attention mechanism formula to get intermediate output B
        scores = tf.matmul(key, query, transpose_b=True) / tf.sqrt(tf.cast(d, tf.float32))
        output_B = tf.matmul(self.dropout(tf.nn.softmax(scores, axis=-1)), x1)

        # Make the summation of the two intermediate outputs
        output = tf.concat([output_A,output_B],axis=1)

        return output
    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(MutualCrossAttention, self).get_config()
        config.update({
            'dropout_rate': self.dropout_rate,
            'hidden_dim': self.hidden_dim
        })
        return config


def silu(x):
    gate = Sigmoid()(x)
    out = gate * x
    return out,gate

# class SSM(Layer):
#     def __init__(self):

class Mambaplus(Layer):
    def __init__(self, dropout_rate,hidden_dim,**kwargs):
        super(Mambaplus, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.dropout = Dropout(dropout_rate)

        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(hidden_dim)
        self.fc3 = Dense(hidden_dim)
        self.conv = Conv1D(filters=100,kernel_size=2)


    def call(self, x, **kwargs):
        x1 = self.fc1(x)
        x1_conv = self.conv(x1)
        x1,_ = silu(x1_conv)

        # SSM
        x1_SSM = Dense(256)(x1)

        x2 = self.fc1(x)
        x2,gate_2 = silu(x2)

        x_gate = (1-gate_2)*x1

        x = x1_SSM*x2
        x = x + x_gate
        x = self.fc3(x)
        return x
    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(Mambaplus, self).get_config()
        config.update({
            'dropout_rate': self.dropout_rate,
            'hidden_dim': self.hidden_dim
        })
        return config

# transformer encoder
## 多头注意力
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (
                self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by number of heads"

        self.wq = layers.Dense(embed_size)
        self.wk = layers.Dense(embed_size)
        self.wv = layers.Dense(embed_size)
        self.dense = layers.Dense(embed_size)

    def call(self, x, mask,**kwargs):
        # x.shape: (batch_size, seq_len, embed_size)
        batch_size = tf.shape(x)[0]

        query = self.wq(x)  # (batch_size, seq_len, embed_size)
        key = self.wk(x)  # (batch_size, seq_len, embed_size)
        value = self.wv(x)  # (batch_size, seq_len, embed_size)

        # Split the embedding into self.num_heads different pieces
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len, head_dim)

        attention = self.scaled_dot_product_attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, head_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))  # (batch_size, seq_len, embed_size)

        output = self.dense(concat_attention)  # (batch_size, seq_len, embed_size)
        return output

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
        """Calculate the attention weights."""
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # (batch_size, num_heads, seq_len, seq_len)

        # Scale matmul_qk
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # Add the mask to zero out padding tokens
        if mask is not None:
            logits += (mask * -1e9)

        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(logits, axis=-1)  # (batch_size, num_heads, seq_len, seq_len)

        output = tf.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, head_dim)

        return output

    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            "embed_size":self.embed_size,
            "head_dim":self.head_dim,
            "num_heads":self.num_heads
        })
        return config

class fusion_module(layers.Layer):
    def __init__(self,**kwargs):
        super(fusion_module,self).__init__()

    def build(self, input_layer):
        self.alpha = self.add_weight(
            name="alpha",
            initializer="uniform",
            trainable=True
        )
        super(fusion_module, self).build(input_layer)
    def call(self,inputs,**kwargs):
        transformer,lstm_out = inputs
        alpha_new = sigmoid(self.alpha)
        # alpha_new = 1
        # 加权融合
        # x = alpha_new * lstm_out + (1.0 - alpha_new) * transformer
        x = tf.concat([alpha_new * lstm_out,(1.0 - alpha_new) * transformer],axis=-1)
        return x

    def get_config(self):
        config = super(fusion_module,self).get_config()

        return config
class transformer_encoder(layers.Layer):
    def __init__(self,num_heads,d_model,dff,**kwargs):
        super(transformer_encoder,self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.mulatt = MultiHeadSelfAttention(embed_size=d_model,num_heads=num_heads)
        self.layernormalization = LayerNormalization(epsilon=1e-6)
        self.fcc =  Sequential(
            [Dense(self.dff,activation='relu'),Dense(self.d_model)]
        )
    def call(self, input, **kwargs):
        attn_out = self.mulatt(input,mask=None)
        x = tf.keras.layers.add([input,attn_out])
        x = self.layernormalization(x)
        # 前馈网络
        # x = Dense(self.dff,activation='relu')(x)
        # x = Dense(self.d_model)(x)
        x = self.fcc(x)

        # 残差连接后跟随层归一化
        x = layers.Add()([input, x])
        out = self.layernormalization(x)
        return out

    def get_config(self):
        # 返回包含所有自定义参数的字典
        config = super(transformer_encoder, self).get_config()
        config.update({
            "dff":self.dff,
            "d_model":self.d_model,
            "num_heads":self.num_heads
        })
        return config


# class CrossAttention(Layer):
#     def __init__(self):
#         super(CrossAttention,self).__init__()
#
#     def call(self,input1,input2):
#
#         return
#
#     def get_config(self):
#         # 返回包含所有自定义参数的字典
#         config = super(CrossAttention, self).get_config()
#         config.update({
#
#         })
#         return config

if __name__ == "__main__":
    import tensorflow as tf

    get_blosum62('./Human dataset/PPA_S_data.csv')
    # 检查TensorFlow是否可以访问GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # 打印TensorFlow检测到的GPU信息
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置TensorFlow以增长方式分配GPU内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("TensorFlow is using GPU.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU is available.")
    # se = SE_Net(64)
    x = np.random.randint(1, 10, size=(3, 11, 16))
    x = tf.constant(x,dtype=tf.float32)
    # out = se(x)
    eca = ECA(64)
    out = eca(x)



