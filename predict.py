import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_gpu_available())
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from tensorflow.keras.models import load_model

from tqdm import tqdm

from methods import SE_Net, plot_history, getMatrixLabel_batch, get_batch_list, cal_metric, transformer_encoder, \
    get_win_feature
import pandas as pd
import numpy as np
from methods import fusion_module

def predict(model_weight_path, test_file_name, savepath, residue, win, embedding_dict, custom_objects=None,
                     ):
    print("Predicting..........")
    model = load_model(model_weight_path, custom_objects=custom_objects)
    test = pd.read_csv(test_file_name, header=None)

    result = []

    batch_list = test
    with tqdm(batch_list, total=len(batch_list), desc="Batches", dynamic_ncols=True) as bar:
        for batch_set in bar:
            x_test2, y_test, _ = getMatrixLabel_batch(batch_set, residue, win)

            test_protT5_feature, y_test = get_win_feature(batch_set, embedding_dict, win=21)

            predict_batch = model.predict_on_batch([x_test2, test_protT5_feature])

            result.append([batch_set[3].split("-")[1],batch_set[3].split("-")[0],predict_batch])

    result = pd.DataFrame(data=result,columns=['position','residue','predict_prob'])
    result.to_csv(savepath + 'result.csv')

def model_test_batch(model_weight_path, test_file_name, residue, win, embedding_dict, custom_objects=None,
                     batch_size=64, mode=''):
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
            x_test2, y_test, _ = getMatrixLabel_batch(batch_set, residue, win)

            test_protT5_feature, y_test = get_win_feature(batch_set, embedding_dict, win=21)

            predict_batch = model.predict_on_batch([x_test2, test_protT5_feature])

            predict = np.concatenate((predict, predict_batch), axis=0)
            y_label = np.concatenate((y_label, y_test), axis=0)
    # print(predict)
    cal_metric(predict, y_label, residue, mode)
if __name__ == '__main__':

    win = 500
    filters = 256
    kernel_size = [10, 10]
    group_num = 4
    ration = 4
    drop_rate = 0.5
    learning_rate = 1e-3
    weight_decay = 0.0001
    epoch = 20
    batch_size = 64
    max_feature = 21

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

    # test
    residue = 'Y'

    model_weight_path = './model/' + residue + '/train_' + residue + ('1_weight_best_2.h5')

    test_file_name = "./data/test_" + residue + "1.csv"
    savepath = "./result/"


    custom_object = {"SE_Net": SE_Net,
                     "transformer_encoder": transformer_encoder,
                     "fusion_module": fusion_module
                     }

    # model_test_batch(model_weight_path, test_file_name, residue, win, embedding_dict, custom_object, batch_size, mode=mode)
    predict(model_weight_path, test_file_name, savepath,residue, win, embedding_dict, custom_object)

