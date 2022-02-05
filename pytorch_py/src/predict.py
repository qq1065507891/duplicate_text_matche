import torch
import time

from torch.utils.data import DataLoader

from src.utils.config import config
from src.utils.process import process_pre_text, process_batch_pre_text
from src.utils.util import get_time_idf, make_seed
from src.model.bert_Q import Question_Bert
from src.utils.dataset import CustomDataset,  collate_fn


def predict(question1, question2):
    """
    预测单个的text
    """
    make_seed(1001)
    print('开始预处理数据')

    start_time = time.time()
    encoding = process_pre_text(question1, question2)
    end_time = get_time_idf(start_time)

    print('处理数据完成， 用时： ', end_time)

    dataset = CustomDataset(encoding)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Question_Bert(config)
    model.load_state_dict(torch.load(config.model_path))

    for batch in dataloader:
        with torch.no_grad():
            pre = model(batch)
            pre = torch.sigmoid(pre).data.tolist()
            pre_f = [0 if pre < 0.5 else 1 for pre in pre]

    print('预测结果', pre_f[0], ',  概率：', pre[0])


def batch_predict(question1, question2):
    """
    预测序列
    """
    make_seed(1001)
    if not isinstance(question1, list) and not isinstance(question2, list):
        raise

    print('开始预处理数据')

    start_time = time.time()
    encoding = process_batch_pre_text(question1, question2)
    end_time = get_time_idf(start_time)

    print('处理数据完成， 用时： ', end_time)

    dataset = CustomDataset(encoding)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Question_Bert(config)
    model.load_state_dict(torch.load(config.model_path))
    all_pre_f = []
    all_pre = []

    for batch in dataloader:
        with torch.no_grad():
            pre = model(batch)
            pre = torch.sigmoid(pre).data.tolist()
            pre_f = [0 if pre < 0.5 else 1 for pre in pre]
            all_pre.append(pre[0])
            all_pre_f.append(pre_f[0])
    print('预测结果', all_pre_f, ',  概率：', all_pre)
