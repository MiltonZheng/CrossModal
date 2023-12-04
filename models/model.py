import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


"""
model for image modality
"""
class ImgNet(nn.Layer):
    def __init__(self, config=None):
        super(ImgNet, self).__init__()
        self.config = config
        self.encode = nn.Sequential(nn.Linear(4096, 1024, bias_attr=False),
                                    nn.BatchNorm(1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.config.code_len, bias_attr=False),
                                    nn.BatchNorm(self.config.code_len))

    def forward(self, x, alpha=1.0):
        hid = self.encode(x)
        code = paddle.tanh(hid * alpha)
        return code, hid

"""
model for text modality
"""
class TxtNet(nn.Layer):
    def __init__(self, config=None):
        super(TxtNet, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.txt_len, 4096, bias_attr=False)
        self.encode = nn.Sequential(nn.Linear(4096, 1024, bias_attr=False),
                                    nn.BatchNorm(1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, self.config.code_len, bias_attr=False),
                                    nn.BatchNorm(self.config.code_len))
    
    def forward(self, x, alpha=1.0):
        feat1 = self.fc1(x)
        feat = F.relu(feat1)
        hid = self.encode(feat)
        code = paddle.tanh(hid * alpha)
        return code, hid


def cal_similarity_high_feature(config, feature_img, feature_text, label=None, alpha=0.5):
    # 计算余弦相似度
    feature_img = F.normalize(feature_img)
    feature_text = F.normalize(feature_text)
    cos_distance_img = paddle.matmul(feature_img, feature_img.t())
    cos_distance_text = paddle.matmul(feature_text, feature_text.t())
    cos_distance_img = cos_distance_img * 2 - 1
    cos_distance_text = cos_distance_text * 2 - 1
    joint = cos_distance_img * alpha + (1 - alpha) * cos_distance_text

    return joint, cos_distance_img, cos_distance_text

def calculate_hamming_dist(outputs1, outputs2):
    ip = paddle.mm(outputs1, outputs2.t())
    mod = paddle.mm((outputs1 ** 2).sum(dim=1).reshape(-1, 1), (outputs2 ** 2).sum(dim=1).reshape(1, -1))
    cos = ip / mod.sqrt()
    hash_bit = outputs1.shape[1]
    dist_ham = hash_bit / 2.0 * (1.0 - cos)
    return dist_ham