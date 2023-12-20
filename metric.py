import paddle
import paddle.nn.functional as F
import numpy as np
from tqdm import tqdm

def codeGen(codeNet_I, codeNet_T, test_loader, database_loader):
    re_BI = []
    re_BT = []
    re_L = []
    database_loader = tqdm(database_loader, desc="database")
    for idx, (data_I, data_T, data_L, _) in database_loader:
        data_T = paddle.cast(data_T, "float32")
        with paddle.no_grad():
            hashcode_I, _ = codeNet_I(data_I)
            hashcode_T, _ = codeNet_T(data_T)
        hashcode_I = paddle.sign(hashcode_I)
        hashcode_T = paddle.sign(hashcode_T)
        re_BI.extend(hashcode_I)
        re_BT.extend(hashcode_T)
        re_L.extend(data_L)
        
    qu_BI = []
    qu_BT = []
    qu_L = []
    
    test_loader = tqdm(test_loader, desc="test")
    for idx, (test_I, test_T, test_L,  _) in test_loader:
        test_T = paddle.cast(test_T, "float32")
        with paddle.no_grad():
            hashcode_I, _ = codeNet_I(test_I)
            hashcode_T, _ = codeNet_T(test_T)
        hashcode_I = paddle.sign(hashcode_I)
        hashcode_T = paddle.sign(hashcode_T)
        qu_BI.extend(hashcode_I)
        qu_BT.extend(hashcode_T)
        qu_L.extend(test_L)
        
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

def cal_mAp(re_B, re_L, qu_B, qu_L):
    query_num = len(qu_L)
    retrieval_num = len(re_L)
    mAp = 0.
    for i in range(query_num):
        query_label = qu_L[i]
        query_code = qu_B[i]
        gnd = (np.asarray(re_L) == query_label).astype("float32")
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = np.sum((np.asarray(re_B) != query_code).astype("float32"), axis=1)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.
        mAp += np.mean(count / (tindex))
    mAp /= query_num
    return mAp