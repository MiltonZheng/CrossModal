import argparse

parser = argparse.ArgumentParser(description='CHV')
# model
parser.add_argument('--backbone', type=str, default='vgg19', help='vgg16,resnet,vit')
parser.add_argument('--feature_dim', type=int, default=4096, help='dim of vgg output(4096,2048,768)')

# dataset config
parser.add_argument('--data_name', type=str, default='mirflickr', help='WIKI or mirflickr or nuswide')
parser.add_argument('--data_path', type=str, default='/root/CrossModal/data', help='dataset path...')
parser.add_argument('--txt_len', type=int, default=1386, help='the length of text feature')

parser.add_argument('--Tanh', action='store_true', help='use tanh to binarize the hash code')

parser.add_argument('--lr_I', type=float, default=0.0003, help='learning rate for image net')
parser.add_argument('--lr_T', type=float, default=0.0003, help='learning rate for text net')
parser.add_argument('--epochs', type=int, default=100, help='training epoch')
parser.add_argument('--batch_size', type=int, default=256, help='the batch size for training')  # batch_size most be even in this project
parser.add_argument('--eval_epochs', type=int, default=5)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--workers', type=int, default=0, help='number of data loader workers.')
# Hashing
parser.add_argument('--code_len', type=int, default='64', help='hash bit,it can be 8, 16, 32, 64, 128...')
parser.add_argument('--cluster_num', type=int, default=30, help='cluster num')
parser.add_argument('--knn_num', type=int, default=20, help='cluster num')
parser.add_argument('--gamma', type=float, default=2.0, help='scale parameter for Cauchy distribution ')
parser.add_argument('--beta', type=float, default=1.0, help='compression rate for conditional information bottleneck')
parser.add_argument('--alpha', type=float, default=0.5, help='confidence for image ')
parser.add_argument('--R', type=int, default=18000, help='MAP@R')
parser.add_argument('--T', type=float, default=0, help='Threshold for binary')
# Loss
