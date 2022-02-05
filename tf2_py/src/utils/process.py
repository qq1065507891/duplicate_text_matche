import os

from sklearn.model_selection import train_test_split
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

from src.utils.config import config
from src.utils.utils import read_file, ensure_dir, save_pkl, load_pkl

config = config


def enconde_data(q1, q2, label=None):
    """
    将句子转为bert输入
    :return:
    """
    assert len(q1) == len(q2)

    tokenizer = Tokenizer(config.dict_path, do_lower_case=True)
    token_ids, segment_ids, labels = [], [], []
    for i in range(len(q1)):
        token_id, segment_id = tokenizer.encode(str(q1[i]), str(q2[i]), maxlen=config.max_len)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
        if label:
            int_label = int(label[i])
            labels.append([int_label])

    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    if labels:
        labels = sequence_padding(labels)
        return token_ids, segment_ids, labels
    else:
        return token_ids, segment_ids


def process_text():
    """
    预处理数据
    :return:
    """
    if os.path.exists(config.train_out_path):
        train = load_pkl(config.train_out_path, 'train')
        test = load_pkl(config.test_out_path, 'test')
    else:
        question1, question2, is_duplicate = read_file(config.file_path)
        q1_train, q1_val, q2_train, q2_val, train_label, test_label = train_test_split(question1, question2,
                                                                                       is_duplicate, test_size=0.2,
                                                                                       stratify=is_duplicate)
        train, test = {}, {}
        train_token_ids, train_segment_ids, train_labels = enconde_data(q1_train, q2_train, train_label)
        test_token_ids, test_segment_ids, test_labels = enconde_data(q1_val, q2_val, test_label)

        ensure_dir(config.out_path)

        train['token_ids'] = train_token_ids
        train['segment_ids'] = train_segment_ids
        train['label'] = train_labels

        test['token_ids'] = test_token_ids
        test['segment_ids'] = test_segment_ids
        test['label'] = test_labels

        save_pkl(config.train_out_path, train, 'train', use_bert=True)
        save_pkl(config.test_out_path, test, 'test', use_bert=True)

    return train, test


def process_pre_text(q1, q2):
    """
    处理预测的文本
    :param q1:
    :param q2:
    :return:
    """
    assert type(q1) == type(q2)
    if not isinstance(q1, list) and not isinstance(q2, list):
        q1, q2 = list(q1), list(q2)
    token_ids, segment_ids = enconde_data(q1, q2)
    return token_ids, segment_ids

