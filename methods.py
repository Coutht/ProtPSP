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
    for i in tqdm(range(len(negative_batch))):

        batch_set = pd.concat([positive_batch[i%len(positive_batch)], negative_batch[i]]).sample(frac=1)
        batch_set_list.append(batch_set)
    return batch_set_list


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




def cal_metric(y_pred,y_true,residue,mode):
    np.set_printoptions(threshold=sys.maxsize)
    dataset = 'data/'
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
def BahdanauAttention(query,value,hidden_unit):

    att_w1 = Dense(hidden_unit)
    att_w2 = Dense(hidden_unit)
    att_v = Dense(1)
    query = tf.expand_dims(query,axis=1) # none,1,256
    score = att_v(tf.nn.tanh(att_w1(query) + att_w2(value))) # none,21,1
    weight = tf.nn.softmax(score,axis=1) # none,21,1
    value_transpose = tf.transpose(value,perm=[0,2,1]) # none,256,21
    context = tf.matmul(value_transpose,weight) # none,256,1
    context = tf.transpose(context,perm=[0,2,1])# none,1,256
    return context,weight

def SE_net(input,units,ration):
    squeeze_1 = GlobalAveragePooling1D()(input)

    excitation_1 = Dense(units // ration,
                         activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-5))(squeeze_1)
    excitation_1 = Dense(units,
                         activation='sigmoid',
                         kernel_initializer='he_normal',
                         bias_initializer='zeros',
                         kernel_regularizer=l2(1e-5))(excitation_1)

    x_weighted = tf.multiply(input, tf.expand_dims(excitation_1, axis=1))

    return x_weighted


def get_feature(batch_set):
    protT5_feature = []
    labels = []

    for i in range(len(batch_set)):
        label = batch_set.iloc[i,0]
        feature = np.array(batch_set.iloc[i,-1][1:-1].split(","),dtype=float)
        protT5_feature.append(feature)
        labels.append(label)

    return np.array(protT5_feature), kutils.to_categorical(labels)

def get_win_feature(batch_set,embedding_dict,win=21):
    protT5_feature = []
    labels = []
    half_len = win//2
    for i in range(len(batch_set)):
        id = batch_set.iloc[i,1]
        seq_len = int(batch_set.iloc[i,2])
        last_embedding = embedding_dict[id]

        # seq = batch_set.iloc[i,-2]
        # seq_len = len(seq)
        position = int(batch_set.iloc[i,3].split("-")[1])
        left = max(position - 1 - half_len, 0)
        right = min(position + half_len, seq_len)
        origin_feature = last_embedding[left:right]
        pad_left=0
        pad_right=0
        if(position-half_len)<=0:
            pad_left = half_len-position + 1
        if(position+half_len-seq_len)>0:
            pad_right = position+half_len-seq_len

        short_embedding = np.vstack([np.zeros((pad_left, 1024)), origin_feature, np.zeros((pad_right, 1024))])
        # feature = batch_set.iloc[i,-1]
        label = batch_set.iloc[i,0]
        # feature = short_embedding.cpu().numpy()  用于pytorch中将张量转移至cpu
        feature = short_embedding
        protT5_feature.append(feature)
        labels.append(label)

    return np.array(protT5_feature), kutils.to_categorical(labels)
