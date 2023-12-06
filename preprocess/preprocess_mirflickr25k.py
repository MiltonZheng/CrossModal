import os
import scipy.io
import numpy as np


QUERY_SIZE = 2000
TRAINING_SIZE = 10000
DATABASE_SIZE = 18015

"""
数据集来自https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_tensorflow/DCMH_tensorflow
1. lall (20015, 24) 标签的one-hot编码
2. fall (20015, 1) im0.jpg, im1.jpg, ...
3. yall (20015, 1386) 文本模态，筛选出来的描述词个数可能比原始数据集txt文件里面少，因为预处理时只选出了出现在20张图以上的描述词，
这不仅可能导致描述词变少，而且可能使得有些图片没有描述词，这一类图片被清理掉了，这就是为什么cleared_set里面只有20015张图片的原因
4. iall (20015, 224, 224, 3) 图像模态
"""
def split_data(images, tags, labels):
	split_shape = [QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE]
	for i in range(len(split_shape)):
		split_shape[i] = split_shape[i] if i == 0 else split_shape[i] + split_shape[i - 1]
	image_split = {}
	tags_split = {}
	labels_split = {}
	image_split['test'] = images[0:QUERY_SIZE,:]
	image_split['train'] = images[QUERY_SIZE:QUERY_SIZE + TRAINING_SIZE,:]
	image_split['database'] = images[QUERY_SIZE:QUERY_SIZE + DATABASE_SIZE,:]
 
	tags_split['test'] = tags[0:QUERY_SIZE,:]
	tags_split['train'] = tags[QUERY_SIZE:QUERY_SIZE + TRAINING_SIZE,:]
	tags_split['database'] = tags[QUERY_SIZE:QUERY_SIZE + DATABASE_SIZE,:]
 
	labels_split['test'] = labels[0:QUERY_SIZE,:]
	labels_split['train'] = labels[QUERY_SIZE:QUERY_SIZE + TRAINING_SIZE,:]
	labels_split['database'] = labels[QUERY_SIZE:QUERY_SIZE + DATABASE_SIZE,:]
 
	return image_split, tags_split, labels_split


if __name__ == "__main__":
    dataset_path = "E:\CrossModal\data\mirflickr"

	# 读取数据
    label = scipy.io.loadmat(os.path.join(dataset_path, 'mirflickr25k-lall.mat'))["LAll"]
    tag = scipy.io.loadmat(os.path.join(dataset_path, 'mirflickr25k-yall.mat'))["YAll"]
    image = np.load(os.path.join(dataset_path, 'image_vgg19_feature.npy')) # NCHW (or NCWH?)

    images, txts, labels = split_data(image, tag, label)

	# 保存数据
    for set in ["test", "train", "database"]:
        np.save(os.path.join(dataset_path, f"{set}_img.npy"), images[set])
        np.save(os.path.join(dataset_path, f"{set}_text.npy"), txts[set])
        np.save(os.path.join(dataset_path, f"{set}_label.npy"), labels[set])
