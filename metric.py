import paddle
from utils import calculate_hamming_dist

def codeGen(codeNet_I, codeNet_T, query_loader, retrieval_loader):
    re_BI = []
    re_BT = []
    re_L = []
    for idx, (data_I, data_T, data_L, _) in enumerate(retrieval_loader):
        data_T = paddle.cast(data_T, "float32")
        data_L = paddle.cast(data_L, "float32")
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
    
    for idx, (test_I, test_T, test_L,  _) in enumerate(query_loader):
        test_T = paddle.cast(test_T, "float32")
        test_L = paddle.cast(test_L, "float32")
        with paddle.no_grad():
            hashcode_I, _ = codeNet_I(test_I)
            hashcode_T, _ = codeNet_T(test_T)
        hashcode_I = paddle.sign(hashcode_I)
        hashcode_T = paddle.sign(hashcode_T)
        qu_BI.extend(hashcode_I)
        qu_BT.extend(hashcode_T)
        qu_L.extend(test_L)
    
    re_BI = paddle.to_tensor(re_BI)
    re_BT = paddle.to_tensor(re_BT)
    re_L = paddle.to_tensor(re_L)
    qu_BI = paddle.to_tensor(qu_BI)
    qu_BT = paddle.to_tensor(qu_BT)
    qu_L = paddle.to_tensor(qu_L)
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def cal_mAP(qu_B, qu_L, re_B, re_L, k):
    query_num = qu_L.shape[0]
    mAP, precision, recall = 0., 0., 0.
    for i in range(query_num):
        gnd = (paddle.mm(qu_L[i, :], re_L.t()) > 0).astype(paddle.float32)
        hamm = calculate_hamming_dist(qu_B[i, :], re_B)
        ind = paddle.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[:k]
        tsum = paddle.sum(tgnd)
        if tsum == 0:
            continue
        count = paddle.linspace(1, tsum, int(tsum))
        tindex = paddle.to_tensor(paddle.where(tgnd == 1)) + 1.
        tindex = paddle.reshape(tindex, [-1])
        mAP += paddle.mean(count / (tindex))
        precision += tsum / k
        recall += tsum / paddle.sum(gnd)
    mAP /= query_num
    precision /= query_num
    recall /= query_num
    return mAP, precision, recall