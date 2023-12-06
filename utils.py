import paddle
import paddle.nn.functional as F

def cal_knn(config, code_I, code_T, labels=None, k=None):
    # 归一化
    code_I = F.normalize(code_I)
    code_T = F.normalize(code_T)
    
    # 计算余弦相似度
    # 例如sim_I[i][j]就是第i和第j张图像哈希码的余弦相似度
    # 越接近1越相似，对角都是1
    sim_I = paddle.mm(code_I, paddle.t(code_I))
    sim_T = paddle.mm(code_T, paddle.t(code_T))
    
    # 计算不在top-k范围内的相似关系数量    
    if k == None:
        k = config.knn_num
    total = code_I.shape[0]
    bottom = total - k - 1
    
    # 按照相似度递增的顺序返回对应的索引
    id_I = sim_I.argsort()
    id_T = sim_T.argsort()
    
    sim_I[paddle.reshape(paddle.arange(total), [-1,1]), id_I[:, :bottom]] = 0
    sim_I[paddle.arange(total),paddle.reshape(id_I[:, -1:],[-1])]=0
    
    sim_T[paddle.reshape(paddle.arange(total), [-1,1]), id_T[:, :bottom]] = 0
    sim_T[paddle.arange(total),paddle.reshape(id_T[:, -1:],[-1])]=0
    
    pass