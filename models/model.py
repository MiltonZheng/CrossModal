import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils import calculate_hamming_dist


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


def contrastive_loss(laplacian, laplacian_I, laplacian_T, img_hashcode, txt_hashcode, gamma=2.0, alpha=0.3):
    dist_img = calculate_hamming_dist(img_hashcode, img_hashcode)
    dist_txt = calculate_hamming_dist(txt_hashcode, txt_hashcode)
    dist_img_txt = calculate_hamming_dist(img_hashcode, txt_hashcode)
    
    # 越相似，对应值就越大，例如自己和自己dist=0，那么sim=1
    sim_I = gamma / (gamma + dist_img)
    sim_T = gamma / (gamma + dist_txt)
    sim_I_T = gamma / (gamma + dist_img_txt)
    
    # intra_I为文本中topk样本对所对应的图像的相似度
    # 由于取了负值以及laplacian的负值，因此topk相似度越高，intra_I的值也就越高
    intra_I = -laplacian_T[laplacian_T < 0] * sim_I[laplacian_T < 0]
    inter_I = sim_I[laplacian_T == 0]
    
    intra_T = -laplacian_I[laplacian_I < 0] * sim_T[laplacian_I < 0]
    inter_T = sim_T[laplacian_I == 0]
    
    # 取负之后，同一个样本的文本图像哈希码就会纳入intra的计算，所以训练会促使同一样本的不同模态哈希码靠近
    laplacian_I[laplacian_I > 0.0] = -1.0
    laplacian_T[laplacian_T > 0.0] = -1.0
    
    intra_I_T_1 = -laplacian_T[laplacian_T < 0] * sim_I_T[laplacian_T < 0]
    inter_I_T_1 = sim_I_T[laplacian_T == 0]
    
    intra_I_T_2 = -laplacian_I[laplacian_I < 0] * sim_I_T[laplacian_I < 0]
    inter_I_T_2 = sim_I_T[laplacian_I == 0]
    
    intra_I_T_sum = alpha * intra_I_T_1.sum() + (1 - alpha) * intra_I_T_2.sum()
    inter_I_T_sum = alpha * inter_I_T_1.sum() + (1 - alpha) * inter_I_T_2.sum()
    
    # 目标是让topk的相似度尽可能高，其他的尽可能低
    loss = -0.1 * paddle.log(paddle.sum(intra_I) / paddle.sum(inter_I)) \
        - 0.1 * paddle.log(paddle.sum(intra_T) / paddle.sum(inter_T)) \
            - paddle.log(paddle.sum(intra_I_T_sum) / paddle.sum(inter_I_T_sum))
        
    return loss
