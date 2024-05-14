from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torchvision.datasets import MNIST
dataset = MNIST('datasets', download=True)

from utils import initialize
class MyMNIST(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Lambda(lambda t: t.float() / 255),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i].unsqueeze(0)  # shape 28x28 => 1x28x28
        return {'pixel_values': self.transform(x)}

def get_data_loader(batch_size):
    dataset = MNIST('datasets', download=True)
    dataset = MyMNIST(dataset.data)
    mnist_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
    return mnist_dataloader