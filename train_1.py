import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_gpu_available())
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from tensorflow.keras.models import load_model
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Embedding,Conv1D,ReLU,BatchNormalization,
                          Dropout,LSTM,Bidirectional,Dense,Softmax,LayerNormalization,GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tqdm import tqdm

from methods import SE_Net, plot_history, getMatrixLabel_batch, get_batch_list, cal_metric, transformer_encoder, SE_net, \
    BahdanauAttention, get_win_feature
import tensorflow.keras.utils as kutils
import pandas as pd
import numpy as np
from methods import fusion_module





def get_model(win,max_features,filters,kernel_size,group_num,drop_rate,ration,learning_rate=1e-3,weight_decay=1e-5):

    input2 = Input(shape=(win,))

    # Embedding
    x_emb2 = Embedding(max_features, 16)(input2)

    # position embedding
    len = input2.shape[-1]
    pos2 = tf.range(len, dtype=tf.int32)
    pos2 = tf.expand_dims(pos2, 0)
    pos2 = tf.tile(pos2, [tf.shape(input2)[0], 1])
    x_posemb2 = Embedding(win, 16)(pos2)

    x2 = x_emb2 + x_posemb2  # none,21,16
    x2 = LayerNormalization(epsilon=1e-6)(x2)

    # group conv
    x2_conv = Conv1D(filters=filters, kernel_size=kernel_size[1], groups=group_num, padding="same")(x2)  # none,21,256
    x2 = ReLU()(x2_conv)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(drop_rate)(x2)

    # SEnet
    x2 = SE_net(x2, units=x2.shape[-1], ration=ration)
    units = x2.shape[-1]
    # BiLSTM+Transformer
    transformer_2 = transformer_encoder(num_heads=2, d_model=256, dff=512)(x2)
    lstm_out2, h_n_f2, c_n_f, h_n_b2, c_n_b = Bidirectional(LSTM(units // 2, return_sequences=True, return_state=True))(x2)

    x2 = fusion_module()([transformer_2, lstm_out2])

    context2, weight2 = BahdanauAttention(x2[:,-1,:], x2, hidden_unit=10)

    x2 = tf.reduce_mean(context2, axis=1)


    # 使用win*1024的ProtT5特征
    prot_T5 = Input(shape=(21,1024))
    # 使用卷积层对1024的维度进行降维
    x_t5 = Conv1D(filters=256,kernel_size=10,padding='same')(prot_T5)
    x_t5 = Dropout(drop_rate)(x_t5)
    # x_t5 = BatchNormalization()(x_t5)
    x_t5 = Conv1D(filters=64,kernel_size=7,padding='same')(x_t5)
    x_t5 = Dropout(drop_rate)(x_t5)
    # x_t5 = BatchNormalization()(x_t5)
    x_t5 = Conv1D(filters=16,kernel_size=3,padding='same')(x_t5)
    x_t5 = LayerNormalization(epsilon=1e-6)(x_t5)
    x_t5 = Dropout(drop_rate)(x_t5)

    # group conv
    x_t5_conv = Conv1D(filters=filters, kernel_size=kernel_size[1],groups=group_num, padding="same")(x_t5)  # none,21,256
    x_t5 = ReLU()(x_t5_conv)
    x_t5 = BatchNormalization()(x_t5)
    x_t5 = Dropout(drop_rate)(x_t5)
    #
    # SEnet
    x_t5 = SE_net(x_t5, units=x_t5.shape[-1], ration=ration)
    units = x_t5.shape[-1]
    # BiLSTM
    transformer_t5 = transformer_encoder(num_heads=2, d_model=256, dff=512)(x_t5)
    lstm_out_t5, h_n_f_t5, c_n_f, h_n_b_t5, c_n_b = Bidirectional(LSTM(units // 2, return_sequences=True, return_state=True))(x_t5)


    x_t5 = fusion_module()([transformer_t5,lstm_out_t5])

    context_t5, weight_t5 = BahdanauAttention(x_t5[:,-1,:], x_t5, hidden_unit=10)
    x_t5 = tf.reduce_mean(context_t5, axis=1)

    x = tf.concat([x2,x_t5],axis=1)

    x = Dense(32,activation="relu")(x)
    x = Dropout(drop_rate)(x)
    x = Dense(2)(x)
    out = Softmax()(x)

    model = Model(inputs=[input2,prot_T5], outputs=out,name="protpsp")
    model.summary()
    opt = Adam(learning_rate=learning_rate, decay=weight_decay)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy', 'AUC', 'Recall'])

    return model


def train_batch(train_file_name,residue,epoch,batch_size,win, max_features,filters,kernel_size,group_num,ration,embedding_dict,mode,drop_rate=0.2,weight_decay = 0.0001,learning_rate = 0.001):

    save_dictory = train_file_name.split('/')[1] + mode

    if not os.path.exists("./model/" + save_dictory + residue):
        os.makedirs("./model/" + save_dictory + residue, exist_ok=True)

    weight_name = train_file_name.split("/")[-1].split(".")[0] + "_weight"
    model_savepath = "./model/" + save_dictory + residue + '/' + weight_name + '_best_2.h5'


    model = get_model(win, max_features,filters,kernel_size,group_num,drop_rate,ration,learning_rate,weight_decay)



    train = pd.read_csv(train_file_name, header=None)

    # batch_size = 32
    positive = train[train[0] == 1]
    negative = train[train[0] == 0]
    positive_val = positive.sample(frac=0.2)
    negative_val = negative.sample(n=len(positive_val))
    # train set
    positive = positive.drop(positive_val.index)
    negative = negative.drop(negative_val.index)


    print('Train...')
    best_loss = 2000
    hist = pd.DataFrame(data=[],columns=["epoch","loss","accuracy","auc","val_loss","val_accuracy","val_auc"])

    for epoch in range(epoch):
        current_loss = []
        print(f"Epoch {epoch}:")

        # 打乱训练集
        positive = positive.sample(frac=1)
        negative = negative.sample(frac=1)

        batch_list = get_batch_list(positive,negative,int(batch_size/2))
        with tqdm(batch_list,total= len(batch_list), desc="Batches",ncols=120) as bar:
            for batch_set in bar:

                x_train_2, y_train,_ = getMatrixLabel_batch(batch_set, residue, win)

                protT5_feature,y_train = get_win_feature(batch_set,embedding_dict,win=21)

                history=model.train_on_batch([x_train_2,protT5_feature],y_train)

                current_loss.append(history[0])
                bar.set_postfix(loss=f" {history[0]:.4f}",acc=f"{history[1]:.4f}" ,auc=f"{history[2]:.4f}")

        # 计算当前epoch的平均损失
        avg_loss = np.mean(current_loss)
        # validation
        # 手动验证
        val_batch_list = get_batch_list(positive_val, negative_val, int(batch_size / 2))
        val_history = [0*len(history)]
        with tqdm(val_batch_list, total=len(val_batch_list), desc="Batches", ncols=120) as val_bar:
            for val_batch_set in val_bar:

                x_val2, y_val,_ = getMatrixLabel_batch(val_batch_set, residue, win)

                val_protT5_feature,y_val  = get_win_feature(val_batch_set,embedding_dict,win=21)
                val_batch_history=model.test_on_batch([x_val2,val_protT5_feature],y_val)
                val_history = np.add(val_history,val_batch_history)
        val_history = val_history/len(val_batch_list)

        print(f"val_loss:{val_history[0]:.4f} - val_acc:{val_history[1]:.4f} - val_auc:{val_history[2]:.4f} - val_recall:{val_history[3]:.4f}")
        hist.loc[len(hist)] = [epoch,avg_loss,history[1],history[2],val_history[0],val_history[1],val_history[2]]

        # 学习率衰减
        if epoch%2 == 1:
            new_lr = model.optimizer.learning_rate * 0.8
            model.optimizer.learning_rate.assign(new_lr)
            print(f"learning_rate reduce to {new_lr}")
        # if val_history[0] - hist.iloc[len(hist)-1,4] > 0:
        #     new_lr = model.optimizer.learning_rate * 0.5
        #     model.optimizer.learning_rate.assign(new_lr)
        #     print(f"learning_rate reduce to {new_lr}")
        if val_history[0] < best_loss:
            print(f"best model saved in {model_savepath}")
            model.save(model_savepath)
            best_loss = val_history[0]

        model_savepath_epoch = "./model/" + save_dictory + residue + '/' + weight_name + '_' + str(epoch) + '.h5'
        model.save(model_savepath_epoch)
    plot_history(hist, save_dictory, residue)

    model_savepath1 = "./model/" + save_dictory + residue + '/' + weight_name + '_finally.h5'
    model.save(model_savepath1)

if __name__ == '__main__':

    residue = 'Y'

    train_file_name = "./data/train_" + residue + "1.csv"


    win = 500
    filters = 256
    kernel_size = [10,10]
    group_num = 4
    ration = 4
    drop_rate = 0.5
    learning_rate = 1e-3
    weight_decay = 0.0001
    epoch = 20
    batch_size = 64
    max_feature = 21

    # mode = '/Uni_PELM_PSP_dbPTM/ablation/time_SEnet/'
    mode = ''



    print('ProtT5 embedding feature loading...')
    embed_file = np.load("./embedding_dict.npz")
    embedding_dict = {}
    for key in tqdm(embed_file.files):
        embedding_dict[key] = embed_file[key]
    embed_file2 = np.load("./embedding_dict_2.npz")
    embedding_dict2 = {}
    for key in tqdm(embed_file2.files):
        embedding_dict2[key] = embed_file2[key]
    embedding_dict.update(embedding_dict2)

    train_batch(train_file_name,residue,epoch,batch_size,win,max_feature,filters,kernel_size,group_num,ration,embedding_dict,mode,drop_rate,weight_decay,learning_rate)
