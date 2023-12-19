import paddle
import paddle.nn.functional as F
from tqdm import tqdm
import numpy as np

def cal_knn(config, code_I, code_T, labels=None, k=None):
    # 归一化
    code_I = F.normalize(code_I)
    code_T = F.normalize(code_T)
    
    # 计算余弦相似度
    # 例如sim_I[i][j]就是第i和第j张图像哈希码的余弦相似度
    # 越接近1越相似，对角都是1
    sim_I = paddle.mm(code_I, paddle.t(code_I))
    sim_T = paddle.mm(code_T, paddle.t(code_T))
    
    # should've used paddle directly, but it's way too slow, I don't know why
    sim_I = np.asarray(sim_I)
    sim_T = np.asarray(sim_T)
    
    # 计算不在top-k范围内的相似关系数量    
    if k == None:
        k = config.knn_num
    total = code_I.shape[0]
    bottom = total - k - 1
    
    # 按照相似度递增的顺序返回对应的索引
    id_I = sim_I.argsort()
    id_T = sim_T.argsort()
    
    # 将相似度矩阵中的不在top-k范围内、以及对角线上（自己和自己）的相似关系置为0
    # 并且top-k范围内的置为1
    sim_I[np.arange(total).reshape([-1,1]), id_I[:, :bottom]] = 0.
    sim_I[np.arange(total), id_I[:, -1:].reshape(-1)] = 0.
    sim_I[sim_I != 0.] = 1.
    
    sim_T[np.arange(total).reshape([-1,1]), id_T[:, :bottom]] = 0.
    sim_T[np.arange(total), id_T[:, -1:].reshape(-1)] = 0.
    sim_T[sim_T != 0.] = 1.
    # sim_T[paddle.reshape(paddle.arange(total), [-1,1]), id_T[:, :bottom]] = 0.
    # sim_T[paddle.arange(total), paddle.reshape(id_T[:, -1:],[-1])] = 0.
    
    sim = sim_I + sim_T
    sim[sim > 0.] = 1.
    
    sim = paddle.to_tensor(sim)
    sim_I = paddle.to_tensor(sim_I)
    sim_T = paddle.to_tensor(sim_T)
    
    return sim, sim_I, sim_T


def generate_laplacian_matrix(sim, eta=1.0):
    D_ = F.diag_embed(1. / paddle.sqrt(paddle.sum(sim > 0, axis=1).reshape([1, -1]).astype("float32"))).reshape([sim.shape[0], -1])
    P = D_ @ sim @ D_
    P_2 = P @ P.t()
    I = paddle.eye(P.shape[0])
    L = I - P
    return L