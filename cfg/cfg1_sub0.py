import torch

class cfg:
    EXP_NO = "1_sub0"
    model_name ='seresnext26d_32x4d'#"resnet34d"
    seed = 48
    folds = 5
    epoch = 20
    batch_size = 50
    lr = 1.0e-03
    enable_amp = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    duration = 2000
    weight_decay = 1.0e-02
    early_stop = 10
    kernel_sizes =(32, 16, 2)
