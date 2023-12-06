import paddle
import numpy as np
import scipy.io as scio
from paddle.io import Dataset, DataLoader


class DatasetStraight(Dataset):
    def __init__(self, config, data_type='train'):
        assert (data_type in ['train', 'test', 'database']), "data_type should be 'train' or 'test' or 'database'"
        # open the train.npy, test.npy and database.npy for loader
        self.file_img = open(config.data_path + config.data_name + f'\{data_type}_img.npy', 'rb')
        self.file_text = open(config.data_path + config.data_name + f'\{data_type}_text.npy', 'rb')
        self.file_label = open(config.data_path + config.data_name + f'\{data_type}_label.npy', 'rb')
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
    test_dataset = DatasetStraight(config, data_type='test')
    database_dataset = DatasetStraight(config, data_type='database')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.workers,
                              drop_last=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.workers)

    database_loader = DataLoader(dataset=database_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.workers,
                                 drop_last=True)
    
    return train_loader, test_loader, database_loader