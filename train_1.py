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

from methods import SE_Net, plot_history, getMatrixLabel_batch, get_batch_list, cal_metric, transformer_encoder
import tensorflow.keras.utils as kutils
import pandas as pd
import numpy as np
from methods import fusion_module



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



def position_encode(x,d_model):
    max_len = x.shape[1]
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # 按照sin和cos交替
    pos_encoding = np.zeros((max_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.constant(pos_encoding, dtype=tf.float32)

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
def getGlobal_batch(batch_set, sites, window_size=49, empty_aa='*'):
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
            if len(sseq)<window_size:
                pad_len = window_size - len(sseq)
                pad_len = int(pad_len)
                short_seq =  sseq + ''.join([empty_aa for count in range(pad_len)])
            else:
                short_seq = sseq[0:window_size]
            short_seqs.append(short_seq)

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

def get_feature(batch_set):
    protT5_feature = []
    labels = []

    for i in range(len(batch_set)):
        label = batch_set.iloc[i,0]
        feature = np.array(batch_set.iloc[i,-1][1:-1].split(","),dtype=float)
        protT5_feature.append(feature)
        labels.append(label)

    return np.array(protT5_feature), kutils.to_categorical(labels)

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# tokenizer = T5Tokenizer.from_pretrained("./ProtT5", do_lower_case=False)
# model = T5EncoderModel.from_pretrained("./ProtT5").to(device)
# def get_feature_win(batch_set,win):
#     protT5_feature = []
#     labels = []
#     half_len = win//2
#     for i in range(len(batch_set)):
#         seq = batch_set.iloc[i,-2]
#         seq_len = len(seq)
#         position = int(batch_set.iloc[i,3].split("-")[1])
#         new_position = position
#         if(position-half_len)<=0:
#             pad_len = half_len-position + 1
#             seq = pad_len * 'X' + seq
#             new_position = new_position + pad_len
#         if(position+half_len-seq_len)>0:
#             pad_len = position+half_len-seq_len
#             seq = seq + pad_len * 'X'
#         seq = " ".join(list(re.sub(r"[UZOB]","X",seq)))
#         ids = tokenizer(seq,add_special_tokens=True,padding="longest")
#         input_ids = torch.tensor(ids["input_ids"]).to(device).unsqueeze(0)
#         attention_mask = torch.tensor(ids["attention_mask"]).to(device).unsqueeze(0)
#         with torch.no_grad():
#             embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
#             last_embedding = embedding_repr.last_hidden_state[0,:-1]
#         short_embedding = last_embedding[new_position-half_len-1:new_position+half_len]
#         # feature = batch_set.iloc[i,-1]
#         label = batch_set.iloc[i,0]
#         feature = short_embedding.cpu().numpy()
#         protT5_feature.append(feature)
#         labels.append(label)
#
#     return np.array(protT5_feature), kutils.to_categorical(labels)

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
# df = pd.read_csv("./Dataset/Uni_PELM_PSP_dbPTM_CDHIT/train_Y_emb.csv",header=None)
# get_feature(df[0:10],win=291)


def get_Dap_net(win1,win2,win3,max_features,filters,kernel_size,group_num,drop_rate,ration,learning_rate=1e-3,weight_decay=1e-5):
    # input1 = Input(shape=(win1,))
    #
    # # Embedding
    # x_emb = Embedding(max_features,16)(input1)
    #
    # # position embedding
    # len = input1.shape[-1]
    # pos = tf.range(len,dtype=tf.int32)
    # pos = tf.expand_dims(pos,0)
    # pos = tf.tile(pos,[tf.shape(input1)[0],1])
    # x_posemb = Embedding(win1,16)(pos)
    # # x_posemb = position_encode(input1,16)
    # # concat embedding and position embedding
    # x = x_emb + x_posemb  # none,21,16
    # x = LayerNormalization(epsilon=1e-6)(x)
    #
    # # group conv
    # # x_conv = Conv1D(filters=filters,kernel_size=kernel_size[0],groups=group_num,padding="same")(x) # none,21,256
    # x_conv = Conv1D(filters=filters,kernel_size=kernel_size[0],padding="same")(x) # none,21,256
    # x = ReLU()(x_conv)
    # x = BatchNormalization()(x)
    # x = Dropout(drop_rate)(x)
    # #
    # # SEnet
    # x = SE_net(x,units=x.shape[-1],ration=ration)
    # units = x.shape[-1]
    # # BiLSTM
    # transformer_out = transformer_encoder(num_heads=2, d_model=256, dff=512)(x)
    # lstm_out,h_n_f,c_n_f,h_n_b,c_n_b = Bidirectional(LSTM(units//2,return_sequences=True,return_state=True))(x)
    # h_n = tf.concat([h_n_f,h_n_b],axis=-1)
    # # h_n = tf.reshape(h_n,(batch_size,units))
    # # lstm + transformer
    # alpha_raw = tf.Variable(0.5, dtype=tf.float32, trainable=True, name="fusion_alpha_raw")
    # alpha = tf.sigmoid(alpha_raw, name="fusion_alpha")
    # x1 = alpha * lstm_out + (1-alpha) * transformer_out
    #
    # # context,weight = BahdanauAttention(h_n,lstm_out,hidden_unit=10)
    # context,weight = BahdanauAttention(h_n,x1,hidden_unit=10)
    #
    # x1 = tf.reduce_mean(context, axis=1)

    # transformer
    # x1= transformer_encoder(num_heads=1, d_model=256, dff=512)(x)
    # x1_backward = transformer_encoder(num_heads=1, d_model=256, dff=512)(tf.reverse(x,axis=[1]))
    # # x1 = transformer_encoder(num_heads=4, d_model=256, dff=512)(x1)
    # # x1 = transformer_encoder(num_heads=4,d_model=256,dff=512)(x1)
    # x1=tf.concat([x1_forward,x1_backward],axis=-1)
    # h_n = tf.concat([x1_forward[:,-1,:],x1_backward[:,-1,:]],axis=-1)
    # context, weight = BahdanauAttention(h_n, x1, hidden_unit=10)
    # x1 = tf.reduce_mean(x1, axis=1)
    #
    input2 = Input(shape=(win2,))

    # Embedding
    x_emb2 = Embedding(max_features, 16)(input2)

    # position embedding
    len = input2.shape[-1]
    pos2 = tf.range(len, dtype=tf.int32)
    pos2 = tf.expand_dims(pos2, 0)
    pos2 = tf.tile(pos2, [tf.shape(input2)[0], 1])
    x_posemb2 = Embedding(win2, 16)(pos2)

    # x_posemb2 = position_encode(input2,16)
    # concat embedding and position embedding
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
    # x2_pos = position_encode(x2,256)
    # transformer_2 = transformer_encoder(num_heads=2, d_model=256, dff=512)(x2+x2_pos)
    transformer_2 = transformer_encoder(num_heads=2, d_model=256, dff=512)(x2)
    lstm_out2, h_n_f2, c_n_f, h_n_b2, c_n_b = Bidirectional(LSTM(units // 2, return_sequences=True, return_state=True))(x2)
    h_n2= tf.concat([h_n_f2, h_n_b2], axis=-1)
    x2 = fusion_module()([transformer_2, lstm_out2])
    # h_n = tf.reshape(h_n,(batch_size,units))
    context2, weight2 = BahdanauAttention(x2[:,-1,:], x2, hidden_unit=10)

    x2 = tf.reduce_mean(context2, axis=1)


    # 使用预训练蛋白质语言模型提取的特征 只使用中间位点的1024维特征
    # prot_T5 = Input(shape=(1024))
    # x_t5 = Dense(128,activation='relu')(prot_T5)
    # x_t5 = Dropout(drop_rate)(x_t5)
    # x_t5 = Dense(32,activation='relu')(x_t5)
    # x_t5 = Dropout(drop_rate)(x_t5)

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
    # x_t5_conv = Conv1D(filters=filters, kernel_size=3, groups=group_num, padding="same")(x_t5)  # none,21,256
    x_t5_conv = Conv1D(filters=filters, kernel_size=kernel_size[1],groups=group_num, padding="same")(x_t5)  # none,21,256
    x_t5 = ReLU()(x_t5_conv)
    x_t5 = BatchNormalization()(x_t5)
    x_t5 = Dropout(drop_rate)(x_t5)
    #
    # SEnet
    x_t5 = SE_net(x_t5, units=x_t5.shape[-1], ration=ration)
    units = x_t5.shape[-1]
    # BiLSTM
    # x_t5_pos = position_encode(x_t5,256)
    # transformer_t5 = transformer_encoder(num_heads=2, d_model=256, dff=512)(x_t5+x_t5_pos)
    transformer_t5 = transformer_encoder(num_heads=2, d_model=256, dff=512)(x_t5)
    lstm_out_t5, h_n_f_t5, c_n_f, h_n_b_t5, c_n_b = Bidirectional(LSTM(units // 2, return_sequences=True, return_state=True))(x_t5)
    h_n_t5 = tf.concat([h_n_f_t5, h_n_b_t5], axis=-1)
    # h_n = tf.reshape(h_n,(batch_size,units))

    x_t5 = fusion_module()([transformer_t5,lstm_out_t5])
    # alpha_t5_raw = tf.Variable(0.5, dtype=tf.float32, trainable=True, name="fusion_alpha_raw2")
    # alpha_t5 = tf.sigmoid(alpha_t5_raw, name="fusion_alpha")
    # x_t5 = alpha_t5 * lstm_out_t5 + (1 - alpha_t5) * transformer_t5
    context_t5, weight_t5 = BahdanauAttention(x_t5[:,-1,:], x_t5, hidden_unit=10)
    x_t5 = tf.reduce_mean(context_t5, axis=1)

    # transformer
    # x_t5 = transformer_encoder(num_heads=1, d_model=256, dff=512)(x_t5)
    # x_t5_backward = transformer_encoder(num_heads=1, d_model=256, dff=512)(tf.reverse(x_t5,axis=[1]))
    # x_t5 = tf.concat([x_t5_forward,x_t5_backward],axis=-1)
    # h_n_t5 = tf.concat([x_t5_forward[:,-1,:],x_t5_backward[:,-1,:]],axis=-1)
    # context, weight = BahdanauAttention(h_n_t5, x_t5, hidden_unit=10)
    # # x_t5 = tf.reduce_mean(x_t5, axis=1)
    # x_t5 = tf.reduce_mean(x_t5, axis=1)


    # x = tf.concat([x1,x2,x_t5],axis=1)
    x = tf.concat([x2,x_t5],axis=1)
    # x = x2
    # x = x_t5
    x = Dense(32,activation="relu")(x)
    x = Dropout(drop_rate)(x)
    x = Dense(2)(x)
    out = Softmax()(x)

    # model = Model(inputs=[input1,input2,input3], outputs=out,name="Dap_net")
    # model = Model(inputs=[input1,input2], outputs=out,name="Dap_net")
    model = Model(inputs=[input2,prot_T5], outputs=out,name="Dap_net")
    # model = Model(inputs=[input2], outputs=out,name="Dap_net")
    # model = Model(inputs=[prot_T5], outputs=out,name="Dap_net")
    # model = Model(inputs=[input1,input2,prot_T5], outputs=out,name="Dap_net")
    model.summary()
    opt = Adam(learning_rate=learning_rate, decay=weight_decay)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy', 'AUC', 'Recall'])

    return model

# get_Dap_net(21,33,51,21,256,10,4,0.5,4)
def train_batch(train_file_name,residue,epoch,batch_size,win1,win2,win3,max_features,filters,kernel_size,group_num,ration,embedding_dict,mode,drop_rate=0.2,weight_decay = 0.0001,learning_rate = 0.001):
    # train_file_name = "/home/aita/zhiyuan/deepphos/transphos/dataset/PELM_Y_data.csv"

    # save_dictory = train_file_name.split('/')[1] + '/Uni_PELM_PSP_dbPTM/ablation/Dap_net+transformer/'
    save_dictory = train_file_name.split('/')[1] + mode
    # save_dictory = train_file_name.split('/')[1] + '/Uni_PELM_PSP_dbPTM/Dap_net+transformer+lstm+T5-21+2000/'
    # save_dictory = train_file_name.split('/')[1] + '/Uniport/deepphos/'

    if not os.path.exists("./model/" + save_dictory + residue):
        os.makedirs("./model/" + save_dictory + residue, exist_ok=True)

    weight_name = train_file_name.split("/")[-1].split(".")[0] + "_weight"
    model_savepath = "./model/" + save_dictory + residue + '/' + weight_name + '_best_2.h5'


    model = get_Dap_net(win1, win2, win3, max_features,filters,kernel_size,group_num,drop_rate,ration,learning_rate,weight_decay)



    train = pd.read_csv(train_file_name, header=None)

    # batch_size = 32
    positive = train[train[0] == 1]
    negative = train[train[0] == 0]
    positive_val = positive.sample(frac=0.2)
    negative_val = negative.sample(n=len(positive_val))
    # train set
    positive = positive.drop(positive_val.index)
    negative = negative.drop(negative_val.index)

    #validation set

    # val_set = pd.concat([positive_val, negative_val], axis=0, ignore_index=True).sample(frac=1)
    # x_val1, y_val, z = getMatrixLabel_batch(val_set, residue, win1)
    # x_val2, _,_ = getMatrixLabel_batch(val_set, residue, win2)
    # x_val3, _,_ = getMatrixLabel_batch(val_set, residue, win3)
    # val_protT5_feature,y_val  = get_feature(val_set)
    # val_protT5_feature,y_val  = get_win_feature(val_set,embedding_dict,win=21)
    #one hot
    # x_val1 = kutils.to_categorical(x_val1,max_features)
    # x_val2 = kutils.to_categorical(x_val2,max_features)
    # x_val3 = kutils.to_categorical(x_val3,max_features)
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
                # x_train_1, y_train, z = getMatrixLabel_batch(batch_set, residue, win1)
                x_train_2, y_train,_ = getMatrixLabel_batch(batch_set, residue, win2)
                # x_train_2, y_train,_ = getGlobal_batch(batch_set, residue, win2)
                # x_train_3, _,_ = getMatrixLabel_batch(batch_set, residue, win3)
                # protT5_feature,y_train = get_feature(batch_set)
                protT5_feature,y_train = get_win_feature(batch_set,embedding_dict,win=21)
                # one-hot 编码
                # x_train_1 = kutils.to_categorical(x_train_1,max_feature)
                # x_train_2 = kutils.to_categorical(x_train_2,max_feature)
                # x_train_3 = kutils.to_categorical(x_train_3,max_feature)



                history=model.train_on_batch([x_train_2,protT5_feature],y_train)
                # history=model.train_on_batch([x_train_2],y_train)
                # history=model.train_on_batch([protT5_feature],y_train)

                current_loss.append(history[0])
                bar.set_postfix(loss=f" {history[0]:.4f}",acc=f"{history[1]:.4f}" ,auc=f"{history[2]:.4f}")

                # bar.update(1)
                        # print(history)

        # 计算当前epoch的平均损失
        avg_loss = np.mean(current_loss)
        # validation
        # 手动验证
        val_batch_list = get_batch_list(positive_val, negative_val, int(batch_size / 2))
        val_history = [0*len(history)]
        with tqdm(val_batch_list, total=len(val_batch_list), desc="Batches", ncols=120) as val_bar:
            for val_batch_set in val_bar:
                # x_val1, y_val, z = getMatrixLabel_batch(val_batch_set, residue, win1)
                x_val2, y_val,_ = getMatrixLabel_batch(val_batch_set, residue, win2)
                # x_val2, y_val,_ = getGlobal_batch(val_batch_set, residue, win2)
                # x_val3, _,_ = getMatrixLabel_batch(val_set, residue, win3)
                # val_protT5_feature,y_val  = get_feature(val_set)
                val_protT5_feature,y_val  = get_win_feature(val_batch_set,embedding_dict,win=21)
                val_batch_history=model.test_on_batch([x_val2,val_protT5_feature],y_val)
                val_history = np.add(val_history,val_batch_history)
        val_history = val_history/len(val_batch_list)
        # val_history = model.evaluate([x_val1,x_val2,x_val3],y_val)
        # val_history = model.evaluate([x_val1,x_val2],y_val)
        # val_history = model.evaluate([x_val1,val_protT5_feature],y_val)
        # val_history = model.evaluate([x_val1,x_val2,val_protT5_feature],y_val)
        print(f"val_loss:{val_history[0]:.4f} - val_acc:{val_history[1]:.4f} - val_auc:{val_history[2]:.4f} - val_recall:{val_history[3]:.4f}")
        hist.loc[len(hist)] = [epoch,avg_loss,history[1],history[2],val_history[0],val_history[1],val_history[2]]



        # if epoch==1:
        #     new_lr = model.optimizer.learning_rate * 0.1
        #     model.optimizer.learning_rate.assign(new_lr)
        #     print(f"learning_rate reduce to {new_lr}")

        # 学习率衰减
        if val_history[0] - hist.iloc[len(hist)-1,4] > 0:
            new_lr = model.optimizer.learning_rate * 0.5
            model.optimizer.learning_rate.assign(new_lr)
            print(f"learning_rate reduce to {new_lr}")
        if val_history[0] < best_loss:
            print(f"best model saved in {model_savepath}")
            model.save(model_savepath)
            best_loss = val_history[0]

        model_savepath_epoch = "./model/" + save_dictory + residue + '/' + weight_name + '_' + str(epoch) + '.h5'
        model.save(model_savepath_epoch)
    plot_history(hist, save_dictory, residue)
    # model_savepath1 = "./model/" + save_dictory + '/' + residue[0] + '/' + weight_name + '.h5'
    model_savepath1 = "./model/" + save_dictory + residue + '/' + weight_name + '_finally.h5'
    model.save(model_savepath1)

def Dap_net_test_batch(model_weight_path,test_file_name,residue,win1,win2,win3,embedding_dict,custom_objects=None,batch_size=64,mode=''):
    print("Testing..........")
    model = load_model(model_weight_path, custom_objects=custom_objects)
    test = pd.read_csv(test_file_name, header=None)

    # batch_size = 32
    positive = test[test[0] == 1]
    negative = test[test[0] == 0]

    positive = positive.sample(frac=1)
    negative = negative.sample(frac=1)
    predict = np.empty((0, 2))
    y_label = np.empty((0, 2))
    batch_list = get_batch_list(positive, negative, int(batch_size / 2))
    with tqdm(batch_list, total=len(batch_list), desc="Batches", dynamic_ncols=True) as bar:
        for batch_set in bar:
            # x_test1, y_test, z = getMatrixLabel_batch(batch_set, residue, win1)
            x_test2, y_test, _ = getMatrixLabel_batch(batch_set, residue, win2)
            # x_test2, y_test, _ = getGlobal_batch(batch_set, residue, win2)
            # x_test3, _, _ = getMatrixLabel_batch(batch_set, residue, win3)
            # test_protT5_feature,y_test = get_feature(batch_set)
            test_protT5_feature,y_test = get_win_feature(batch_set,embedding_dict,win=21)
            # one_hot
            # x_test1 = kutils.to_categorical(x_test1,max_feature)
            # x_test2 = kutils.to_categorical(x_test2,max_feature)
            # x_test3 = kutils.to_categorical(x_test3,max_feature)

            # predict_batch = model.predict_on_batch([x_test1, x_test2, x_test3])
            # predict_batch = model.predict_on_batch([x_test1, x_test2])
            predict_batch = model.predict_on_batch([x_test2,test_protT5_feature])
            # predict_batch = model.predict_on_batch([x_test2])
            # predict_batch = model.predict_on_batch([test_protT5_feature])
            predict = np.concatenate((predict, predict_batch), axis=0)
            y_label = np.concatenate((y_label, y_test), axis=0)
    # model.summary()
    # predict = model.predict([x_test1,x_test2,x_test3],verbose=1)
    # predict = model.predict(x_test3,verbose=1)
    print(predict)
    cal_metric(predict, y_label, residue, mode)

if __name__ == '__main__':

    residue = 'Y'

    # uniport dataset
    # train_file_name = "./Human dataset/Uniport dataset/CDHIT_0.5/train_" + residue[0] + ".csv"

    # four dataset
    # train_file_name = "./Dataset/Uni_PELM_PSP_dbPTM_CDHIT/train_" + residue + ".csv"
    train_file_name = "./Dataset/Uni_PELM_PSP_dbPTM_CDHIT/train_" + residue + "1.csv"

    win1 = 51
    win2 = 500
    win3 = 51
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
    # mode = '/Uni_PELM_PSP_dbPTM/ablation/Dap_net+transformer+lstm/'
    mode = '/Uni_PELM_PSP_dbPTM/ablation/time_SEnet/'
    # mode = '/Uni_PELM_PSP_dbPTM/ablation/global_500/'

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

    train_batch(train_file_name,residue,epoch,batch_size,win1,win2,win3,max_feature,filters,kernel_size,group_num,ration,embedding_dict,mode,drop_rate,weight_decay,learning_rate)
    # test
    # mode = '/Uniport/deepphos/'

    # mode = '/Uni_PELM_PSP_dbPTM/Dap_net+transformer+lstm+T5-21+2000/'

    # uniport dataset
    # model_weight_path = './model/Human dataset' + mode + residue[0] + '/train_' + residue[0] + ('_weight_best_2.h5')

    # model_weight_path = './model/Dataset' + mode + residue + '/train_' + residue + ('_emb_weight_best_2.h5')
    # model_weight_path = './model/Dataset' + mode + residue + '/train_' + residue + ('1_weight_best_2.h5')
    model_weight_path = './model/Dataset' + mode + residue + '/train_' + residue + ('1_weight_3n.h5')

    # test_file_name = "./Human dataset/Uniport dataset/CDHIT_0.5/test_" + residue[0] + ".csv"

    # test_file_name = "./Dataset/Uni_PELM_PSP_dbPTM_CDHIT/test_" + residue + ".csv"
    test_file_name = "./Dataset/Uni_PELM_PSP_dbPTM_CDHIT/test_" + residue + "1.csv"

    custom_object = {"SE_Net":SE_Net,
                     "transformer_encoder":transformer_encoder,
                     "fusion_module":fusion_module
                     }
    Dap_net_test_batch(model_weight_path,test_file_name,residue,win1,win2,win3,embedding_dict,custom_object,batch_size,mode=mode)
