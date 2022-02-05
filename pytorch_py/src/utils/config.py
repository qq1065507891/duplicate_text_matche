import os

import torch

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]


class Config(object):
    data_path = os.path.join(root_path, 'data/questions.csv')
    model_path = os.path.join(root_path, 'model/bert_model_1.h5')
    model_name = '重复文本匹配模型'

    bert_base_path = os.path.join(root_path, 'bert-base-uncased')

    log_folder_path = os.path.join(root_path, 'log')
    log_path = os.path.join(root_path, 'log.txt')

    train_path = os.path.join(root_path, 'data/train.pkl')
    test_path = os.path.join(root_path, 'data/test.pkl')

    use_gpu = True
    gpu_id = 0

    predict = False

    decay_rate = 0.3
    decay_patience = 5

    hidden_size = 512
    drop_out = 0.5
    batch_size = 16
    epochs = 1
    learning_rate = 2e-5
    require_improvement = 10000


config = Config()
