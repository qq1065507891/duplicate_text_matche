import os

import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from src.utils.config import config
from src.utils.util import save_pkl, load_pkl


def process_text(question1, question2, label):
    """
    处理训练的数据
    :param question1:
    :param question2:
    :param label:
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(config.bert_base_path)

    train, test = [], []
    if os.path.exists(config.train_path):
        train = load_pkl(config.train_path, 'train_data')
        test = load_pkl(config.test_path, 'test_data')
    else:
        q1_train, q1_val, q2_train, q2_val, train_label, test_label = train_test_split(question1, question2, label,
                                                                                       test_size=0.2, stratify=label)
        # train_data = tokenizer.batch_encode_plus(q1_train, q2_train, truncation=True, padding=True, max_length=375)
        # train_encoding.append(train_data['input_ids'])
        # train_encoding.append(train_data['token_type_ids'])
        # train_encoding.append(train_data['attention_mask'])
        #
        # test_data = tokenizer.batch_encode_plus(q1_val, q2_val, truncation=True, padding=True, max_length=375)
        # test_encoding.append(test_data['input_ids'])
        # test_encoding.append(test_data['token_type_ids'])
        # test_encoding.append(test_data['attention_mask'])

        for i in range(len(q1_train)):
            train_encoding = []
            train_data = tokenizer.encode_plus(str(q1_train[i]), str(q2_train[i]), truncation=True,
                                               padding=True, max_length=200)
            train_encoding.append(train_data['input_ids'])
            train_encoding.append(train_data['token_type_ids'])
            train_encoding.append(train_data['attention_mask'])
            train_encoding.append(int(train_label[i]))
            train.append(train_encoding)

        for i in range(len(q1_val)):
            test_encoding = []
            test_data = tokenizer.encode_plus(str(q1_val[i]), str(q2_val[i]), truncation=True,
                                              padding=True, max_length=200)
            test_encoding.append(test_data['input_ids'])
            test_encoding.append(test_data['token_type_ids'])
            test_encoding.append(test_data['attention_mask'])
            test_encoding.append(int(test_label[i]))
            test.append(test_encoding)

        save_pkl(config.train_path, train, 'train_data', use_bert=True)
        save_pkl(config.test_path, test, 'test_data', use_bert=True)

    return train, test


def process_pre_text(question1, question2):
    """
    处理预测数据
    :param question1:
    :param question2:
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(config.bert_base_path)
    train_encoding = []
    train_data = tokenizer.encode_plus(question1, question2, truncation=True, padding=True, max_length=200)
    train_encoding.append(train_data['input_ids'])
    train_encoding.append(train_data['token_type_ids'])
    train_encoding.append(train_data['attention_mask'])
    return [train_encoding]


def process_batch_pre_text(question1, question2):
    """
    处理批量数据
    """
    tokenizer = BertTokenizer.from_pretrained(config.bert_base_path)
    all_data = []
    for i in range(len(question1)):
        train_encoding = []
        train_data = tokenizer.encode_plus(str(question1[i]), str(question2[i]), truncation=True, padding=True,
                                           max_length=200)
        train_encoding.append(train_data['input_ids'])
        train_encoding.append(train_data['token_type_ids'])
        train_encoding.append(train_data['attention_mask'])
        all_data.append(train_encoding)
    return all_data
