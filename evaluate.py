import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from scipy.stats import entropy
import torchvision.transforms as transforms
import logging
class GAN_evaluator:
    def __init__(self, device='cpu', num_classes=1000):
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.device = device
        self.num_classes = num_classes
        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize(299),                              # Resize image to 299x299
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),      # Repeat grayscale channel across RGB channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406],     # Normalize using ImageNet mean
                                std=[0.229, 0.224, 0.225])      # and standard deviation
        ])

    def inception_score(self, images):
        # Set the inception model to evaluation mode
        self.inception_model.eval()
        # Apply the transform to each image in the batch
        images_transformed = torch.stack([self.transform(img) for img in images])
        # Forward pass to get logits
        with torch.no_grad():
            logits = self.inception_model(images_transformed)

        # Convert logits to probabilities
        preds = torch.nn.functional.softmax(logits, dim=1)

        # Calculate marginal distribution
        marginal_dist = preds.mean(dim=0)

        # Calculate IS
        score = np.exp(np.mean([entropy(preds[i].cpu().numpy(), marginal_dist.cpu().numpy()) for i in range(preds.shape[0])]))
        
        return score

    def calculate_fid(self, real_images, generated_images):
        # Set the inception model to evaluation mode
        self.inception_model.eval()
        new_real_images = torch.stack([self.transform(img) for img in real_images])
        new_generated_images = torch.stack([self.transform(img) for img in generated_images])
        # Calculate activations for real and generated images
        with torch.no_grad():
            act1 = self.inception_model(new_real_images)
            act2 = self.inception_model(new_generated_images)

        # Convert tensors to NumPy arrays (assuming act1 and act2 are tensors)
        act1_np = act1.cpu().numpy()
        act2_np = act2.cpu().numpy()

        # Calculate mean and covariance using NumPy
        mu1_np, sigma1_np = np.mean(act1_np, axis=0), np.cov(act1_np, rowvar=False)
        mu2_np, sigma2_np = np.mean(act2_np, axis=0), np.cov(act2_np, rowvar=False)

        # Calculate FID
        ssdiff_np = np.sum((mu1_np - mu2_np) ** 2.0)
        covmean_np = sqrtm(sigma1_np @ sigma2_np)

        # Numerical stability check
        if np.iscomplexobj(covmean_np):
            covmean_np = covmean_np.real

        fid_np = ssdiff_np + np.trace(sigma1_np + sigma2_np - 2.0 * covmean_np)
        return fid_np