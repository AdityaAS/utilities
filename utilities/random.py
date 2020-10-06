import random
import numpy as np
import torch
import os

def seed_everything(seed=0, harsh=False):
    """Sets random seeds for torch, numpy and random and PYTHONHASHSEED
    WARNING: If harsh=True, the code will run slowly

    Args:
        seed (int, optional): Seed initiializer
        harsh (bool, optional): Enables deterministic behaviour for CUDA too
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ['PYTHONHASHSEED'] = str(seed)
