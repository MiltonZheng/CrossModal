import os
import h5py
import numpy as np

import paddle
from paddle.vision import models, transforms
from tqdm import tqdm
from PIL import Image



if __name__ == "__main__":
    dataset_path = '/root/CrossModal/data/mirflickr'
	# 读取数据
    images = h5py.File(os.path.join(dataset_path, 'mirflickr25k-iall.mat'), 'r')["IAll"][()] # NCHW (or NCWH?)

	# 标准化图像数据
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # vgg19模型，替换classifier层，获取4096维特征
    vgg = models.vgg19(pretrained=True)
    new_classifier = paddle.nn.Sequential(*list(vgg.children())[-1][:6])
    vgg.classifier = new_classifier
    # (6): Linear(in_features=4096, out_features=1000, bias=True)
    vgg.eval()
    img_fea = []
    for i in tqdm(range(images.shape[0]), desc="extract feature"):
        # 将 NumPy 数组转换为 PIL 图像
        image = np.transpose(images[i], (2, 1, 0))
        image = Image.fromarray(image)
        transformed_image = trans(image).unsqueeze(0)
        feature = vgg(transformed_image).detach().cpu().numpy()
        img_fea.extend(feature)
		
    image = np.vstack(img_fea)
    np.save(os.path.join(dataset_path, "image_vgg19_feature.npy"), image)

