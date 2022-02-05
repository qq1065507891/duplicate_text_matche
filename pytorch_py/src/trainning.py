import torch
import time
import os
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn import metrics

from src.utils.process import process_text
from src.utils.util import read_file, get_time_idf, make_seed, log
from src.model.bert_Q import Question_Bert
from src.utils.dataset import CustomDataset,  collate_fn
from src.model.focal_loss import BCEFocalLoss


def train_epoch(config, device, train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler=None):
    model.train()

    total_batch = 0  # 记录进行多少batch
    dev_best_loss = float('inf')  # 记录上次最好的验证集loss
    last_improve = 0  # 记录上次提升的batch
    flag = False  # 停止位的标志, 是否很久没提升

    start_time = time.time()

    for epoch in range(config.epochs):
        log.info('Epoch [{}/{}]'.format(epoch + 1, config.epochs))
        for i, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            *x, y = [data.to(device) for data in batch]
            outputs = model(x)

            model.zero_grad()

            loss = loss_fn(outputs, y.float())

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if total_batch % 100 == 0:  # 每训练50次输出在训练集和验证集上的效果
                true = y.data.cpu()
                # predict = torch.max(outputs.data, 1)[1].cpu()
                outputs = outputs.data.cpu()
                predict = [0 if pre < 0.5 else 1 for pre in outputs]
                score = metrics.accuracy_score(true, predict)

                dev_acc, dev_loss = evaluate(config, model, dev_dataloader, device)
                if total_batch > 20000:
                    scheduler.step(dev_loss)

                if dev_best_loss > dev_loss:
                    dev_best_loss = dev_loss

                    torch.save(model.state_dict(), config.model_path)
                    improve = '+'
                    last_improve = total_batch
                else:
                    improve = '-'
                time_idf = get_time_idf(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.6}, Train ACC:{2:>6.2%}, ' \
                      'Val Loss:{3:>5.6}, Val ACC:{4:>6.2%}, Time:{5}  {6}'
                log.info(msg.format(total_batch, loss.item(), score, dev_loss, dev_acc, time_idf, improve))
                model.train()

            total_batch = total_batch + 1

            if total_batch - last_improve > config.require_improvement:
                # 在验证集上loss超过1000batch没有下降, 结束训练
                log.info('在验证集上loss超过10000次训练没有下降, 结束训练')
                flag = True
                break

        if flag:
            break


def traninng(config):
    """
    导入数据，开始训练
    :return:
    """
    make_seed(1001)
    log.info('开始预处理数据')
    start_time = time.time()
    question1, question2, is_duplicate = read_file(config.data_path)
    train_encoding, test_encoding = process_text(question1, question2, is_duplicate)
    end_time = get_time_idf(start_time)
    log.info(f'处理数据完成， 用时： {end_time}')

    train_dataset = CustomDataset(train_encoding)
    dev_dataset = CustomDataset(test_encoding)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn
    )

    if config.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda', config.gpu_id)
    else:
        device = torch.device('cpu')

    model = Question_Bert(config).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = BCEFocalLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                           factor=config.decay_rate,
                                                           patience=config.decay_patience)

    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))

    train_epoch(config, device, train_dataloader, dev_dataloader, model, loss_fn, optimizer, scheduler)


def evaluate(config, model, dev_iter, device, test=False):
    """
    模型评估
    :param config:
    :param model:
    :param dev_iter:
    :param test:
    :return: acc, loss
    """
    model.eval()
    loss_total = 0
    predicts_all = np.array([], dtype=int)
    labels_all = np.array([], dtype='int')

    with torch.no_grad():
        for batch in dev_iter:
            *dev, labels = [data.to(device) for data in batch]
            outputs = model(dev)
            outputs = torch.sigmoid(outputs)
            # loss = F.cross_entropy(outputs, labels)
            outputs = torch.tensor([0 if pre < 0.5 else 1 for pre in outputs], dtype=torch.float).to(device)
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            loss_total += loss
            true = labels.data.cpu().numpy()
            # predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict = outputs.cpu().numpy()
            predicts_all = np.append(predicts_all, predict)
            labels_all = np.append(labels_all, true)

    acc = metrics.accuracy_score(labels_all, predicts_all)

    if test:
        report = metrics.classification_report(labels_all, predicts_all, target_names=config.class_name, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)



