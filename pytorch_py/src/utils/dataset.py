import torch
from torch.utils.data import Dataset

from src.utils.config import config


class CustomDataset(Dataset):
    """
    数据集的类
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        sample = self.data[item]
        return sample

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    lens = [len(data[0]) for data in batch]
    max_len = max(lens)

    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    input_ids = []
    token_type_ids = []
    attention_masks = []
    if not config.predict:
        labels = []
        for data in batch:
            input_id, token_type_id, attention_mask, label = data
            input_ids.append(_padding(input_id, max_len))
            token_type_ids.append(_padding(token_type_id, max_len))
            attention_masks.append(_padding(attention_mask, max_len))
            labels.append(label)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_masks), torch.tensor(labels)
    else:
        for data in batch:
            input_id, token_type_id, attention_mask = data
            input_ids.append(_padding(input_id, max_len))
            token_type_ids.append(_padding(token_type_id, max_len))
            attention_masks.append(_padding(attention_mask, max_len))
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_masks)


