import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)                     # Python 的随机数种子
    np.random.seed(seed)                  # Numpy 的随机数种子
    torch.manual_seed(seed)               # PyTorch 的 CPU 随机数种子
    torch.cuda.manual_seed(seed)          # PyTorch 的 GPU 随机数种子（单个 GPU）
    torch.cuda.manual_seed_all(seed)      # 如果使用多个 GPU，固定所有的 GPU 种子
