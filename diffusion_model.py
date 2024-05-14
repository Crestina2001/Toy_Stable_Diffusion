import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
import matplotlib.pyplot as plt
from utils import extract
from tqdm import tqdm
from utils import initialize


class DiffusionModel:
    def __init__(self, denoise_model, device, params):
        self.timesteps = params.get('timesteps', 200)
        self.beta_start = params.get('beta_start', 0.0001)
        self.beta_end = params.get('beta_end', 0.08)
        self.device = device
        self.denoise_model = denoise_model
        self.channels = params.get('channels', 1)
        self.image_size = params.get('image_size', 28)
        noise_scheduler = params.get('noise_scheduer', 'linear')

        # Initialize betas using the provided sampling method
        self.betas = self.get_beta_schedule(noise_scheduler)
        self.alphas = 1. - self.betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    def get_val(self):
        return (self.betas, self.alphas, self.sqrt_recip_alphas, self.alphas_cumprod, 
                self.alphas_cumprod_prev, self.sqrt_alphas_cumprod, 
                self.sqrt_one_minus_alphas_cumprod, self.posterior_variance)
    
    def cosine_beta_schedule(self, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)

    def quadratic_beta_schedule(self):
        return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.timesteps) ** 2

    def sigmoid_beta_schedule(self):
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start

    def get_beta_schedule(self, sampling_method):
        if sampling_method == 'linear':
            return self.linear_beta_schedule()
        elif sampling_method == 'cosine':
            return self.cosine_beta_schedule()
        elif sampling_method == 'quadratic':
            return self.quadratic_beta_schedule()
        elif sampling_method == 'sigmoid':
            return self.sigmoid_beta_schedule()
        else:
            raise ValueError("Invalid sampling method")
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)
        imgs = [img.cpu().numpy()]

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return torch.tensor(imgs[-1])

    @torch.no_grad()
    def ddpm_sample(self, batch_size=36):
        return self.p_sample_loop(shape=(batch_size, self.channels, self.image_size, self.image_size))

    @torch.no_grad()
    def ddim_sample(self, batch_size=36, ddim_timesteps=50, eta=0.0):
        img = torch.randn((batch_size, self.channels, self.image_size, self.image_size), device=self.device)

        # Create a sequence of timesteps for DDIM
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.array(list(range(0, self.timesteps, c))) + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='DDIM sampling loop'):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=self.device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=self.device, dtype=torch.long)
            
            alpha_cumprod_t = extract(self.alphas_cumprod, t, img.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, img.shape)

            # Predict noise using the denoise model
            pred_noise = self.denoise_model(img, t)
            pred_x0 = (img - torch.sqrt(1. - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)

            # Compute variance and direction for the next step
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * pred_noise

            # Compute x_{t-1}
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigma_t * torch.randn_like(img)
            img = x_prev

        return img





