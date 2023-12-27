import os
import paddle
import numpy as np
from paddle.io import Dataset, DataLoader


class DatasetStraight(Dataset):
    def __init__(self, config, data_type='train'):
        assert (data_type in ['train', 'query', 'retrieval']), "data_type should be 'train' or 'query' or 'retrieval'"
        # open the train.npy, test.npy and database.npy for loader
        dataset_path = os.path.join(config.data_path, config.data_name)
        self.file_img = open(os.path.join(dataset_path, f'{data_type}_img.npy'), 'rb')
        self.file_text = open(os.path.join(dataset_path, f'{data_type}_text.npy'), 'rb')
        self.file_label = open(os.path.join(dataset_path, f'{data_type}_label.npy'), 'rb')
        self.images = np.load(self.file_img)
        self.texts = np.load(self.file_text)
        self.labels = np.load(self.file_label)

    def __getitem__(self, index):
        image = paddle.to_tensor(self.images[index, :])
        text = paddle.to_tensor(self.texts[index, :])
        # to one-hot
        label = paddle.to_tensor(self.labels[index, :])
        return image, text, label, index

    def __len__(self):
        return len(self.images)


def getLoader(config):
    train_dataset = DatasetStraight(config, data_type='train')
    query_dataset = DatasetStraight(config, data_type='query')
    retrieval_dataset = DatasetStraight(config, data_type='retrieval')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.workers,
                              drop_last=True)

    query_loader = DataLoader(dataset=query_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.workers)

    retrieval_loader = DataLoader(dataset=retrieval_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.workers,
                                 drop_last=True)
    
    return train_loader, query_loader, retrieval_loader