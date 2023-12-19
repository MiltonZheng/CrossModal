import math
import paddle
import logging
import numpy as np
from tqdm import tqdm

import config
import utils
from load_data import getLoader
from models.model import ImgNet, TxtNet, contrastive_loss


state = {
    'epoch' : 0,
    
    'best_MAP' : 0.0,
    'best_I2T' : 0.0,
    'best_T2I' : 0.0,
    'best_epoch' : 0,
    'Database_hashpool_path' : None,
    'Testbase_hashpool_path' : None,
    'Trainbase_hashpool_path' : None,
    'final_result' : None,
    'filename_previous_best' : None,
    'iter' : 0,
    'img_feature_last' : None,
    'text_feature_last' : None,
}
logger = None



def train_step(config, codeNet_I, codeNet_T, opt_I, opt_T, train_loader):
    codeNet_I.train()
    codeNet_T.train()
    epoch = state['epoch']
    loss_list = []
    with paddle.no_grad():
        for idx, (img, txt, labels, _) in enumerate(tqdm(train_loader, desc="forward propagation")):
            # 文本以uint8 one-hot格式存储，因此要转换为float32
            txt = paddle.cast(txt, "float32")
            # 提取图像和文本特征
            img_hashcode, feature_img = codeNet_I(img, math.sqrt(1 + epoch))
            text_hashcode, feature_text = codeNet_T(txt, math.sqrt(1 + epoch))
            # 拼接图像和文本特征
            if idx == 0:
                all_img_hashcode, all_text_hashcode = img_hashcode, text_hashcode
            else:
                all_img_hashcode = paddle.concat((all_img_hashcode, img_hashcode), axis=0)
                all_text_hashcode = paddle.concat((all_text_hashcode, img_hashcode), axis=0)
        sim, sim_I, sim_T = utils.cal_knn(config, all_img_hashcode, all_text_hashcode)
        laplacian = utils.generate_laplacian_matrix(sim)
        laplacian_I = utils.generate_laplacian_matrix(sim_I)
        laplacian_T = utils.generate_laplacian_matrix(sim_T)
        
        state['img_feature_last'] = all_img_hashcode
        state['text_feature_last'] = all_text_hashcode
    
    for idx, (img, txt, label, _) in enumerate(train_loader):
        # 文本以uint8 one-hot格式存储，因此要转换为float32
        txt = paddle.cast(txt, "float32")
        opt_I.clear_grad()
        opt_T.clear_grad()
        
        img_hashcode, feature_img = codeNet_I(img, math.sqrt(1 + epoch))
        txt_hashcode, feature_txt = codeNet_T(txt, math.sqrt(1 + epoch))
        
        bs = config.batch_size
        loss_global = contrastive_loss(laplacian[idx*bs:(idx+1)*bs, idx*bs:(idx+1)*bs],
                                        laplacian_I[idx*bs:(idx+1)*bs, idx*bs:(idx+1)*bs],
                                        laplacian_T[idx*bs:(idx+1)*bs, idx*bs:(idx+1)*bs],
                                        img_hashcode, txt_hashcode,
                                        gamma=config.gamma)
        loss = loss_global
        loss_list.append(loss.item())
        loss.backward()
        opt_I.step()
        opt_T.step()
    return np.mean(loss_list)

def train(config):
    train_loader, test_loader, database_loader = getLoader(config)
    codeNet_I = ImgNet(config=config)
    codeNet_T = TxtNet(config=config)
    opt_I = paddle.optimizer.Adam(parameters=codeNet_I.parameters(), learning_rate=config.lr_I)
    opt_T = paddle.optimizer.Adam(parameters=codeNet_T.parameters(), learning_rate=config.lr_T)
    
    for epoch in range(config.epochs):
        state['epoch'] = epoch
        train_step(config, codeNet_I, codeNet_T, opt_I, opt_T, train_loader)
        logger.info(f"epoch: {epoch}")
    pass


if __name__ == "__main__":
    logger = logging.getLogger("training")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    configs = config.parser.parse_args()
    for k, v in vars(configs).items():
        logging.info(f"{k} = {v}")
    
    logger.info("Start training...")
    train(configs)
    pass