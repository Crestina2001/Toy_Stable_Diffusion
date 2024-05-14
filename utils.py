import matplotlib.pyplot as plt
import numpy as np
import torch

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def initialize():
    torch.manual_seed(0)