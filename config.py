class ModelConfig:
    backend = "pytorch"
    
    data_name = "mirflickr" # "mirflickr" or "nuswide" or more
    data_path = f"./data/{data_name}/"
    txt_len_dict = {
        "mirflickr" : 1386
        }
    txt_len = txt_len_dict[data_name]
    code_len = 64 # 16, 32, 64, 128, 256
    
    backbone = "vgg19"
    feature_dim = 4096
    
    lr_I = 3e-4
    lr_T = 3e-4
    epochs = 100
    batch_size = 256
    
    cluster_num = 30
    knn_num = 20
    gamma = 2.0
    beta = 1.0
    alpha = 0.5
    R = 50 # mAP@R
    T = 0 # Threshold for binary
    