import torch
from tqdm.auto import tqdm
from diffusion_model import DiffusionModel
import torch.nn.functional as F
from torchvision.utils import save_image
from utils import initialize
from evaluate import GAN_evaluator
from dataLoader import get_data_loader
import math
import os
import logging
import copy

def p_losses(denoise_model, dif_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    # Use q_sample from the noise_scheduler instance
    x_noisy = dif_model.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def evaluate_model(generated_images, device, batch_size=36):
    # Initialize the data loader to fetch real images
    dataloader = get_data_loader(batch_size)

    # Fetch one batch of real images
    real_images_batch = next(iter(dataloader))
    real_images = real_images_batch["pixel_values"].to(device)

    # Create an instance of GAN_evaluator
    evaluator = GAN_evaluator(device)
    #print(real_images[0][0][0])
    #print(generated_images[0][0][0])

    # Calculate FID score
    fid_score = evaluator.calculate_fid(real_images, generated_images.to(device))

    return fid_score

def train_diffusion_model(denoise_model, dataloader, optimizer, params, device, best_fid):
    # unpack the hyperparams
    results_folder = params.get('results_folder', 'results/')
    num_epochs = params.get('num_epochs', 20)
    sampling_method = params.get('sampling_method', 'DDPM')
    timesteps = params.get('timesteps', 200)
    
    DM = DiffusionModel(denoise_model, device, params)
    best_fid_this_param = float('inf')
    for epoch in range(num_epochs):
        print(f'======= Epoch {epoch + 1}/{num_epochs} =======')
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, batch in progress_bar:
            optimizer.zero_grad()
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = p_losses(denoise_model, DM, batch, t, loss_type="huber")
            progress_bar.set_description(f'Step {step}, Loss: {loss.item():.4f}')
            loss.backward()
            optimizer.step()

        # Choose the sampling method based on the value of sampling_method
        if sampling_method == 'DDIM':
            all_images = DM.ddim_sample(batch_size=36)
        elif sampling_method == 'DDPM':
            all_images = DM.ddpm_sample(batch_size=36)
        else:
            raise ValueError("sampling_method must be either 'DDIM' or 'DDPM'")
        #all_images = torch.tensor(all_images_list[-1])

        fid_score = evaluate_model(all_images, device)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, f'{results_folder}/sample-{epoch+1}.png', nrow=6)
        print(f"epoch {epoch}: fid score: {fid_score}")
        logging.info(f"epoch {epoch}: fid score: {fid_score}")
        if fid_score < best_fid_this_param:
            best_fid_this_param = fid_score
        
    if best_fid_this_param < best_fid:
        # Check if the 'best_results' folder exists, and create it if it doesn't
        best_results_folder = os.path.join(results_folder, 'best_results')
        if not os.path.exists(best_results_folder):
            os.makedirs(best_results_folder)
        # Save the image in the 'best_results' folder
        save_image(all_images, f'{best_results_folder}/sample-{epoch+1}.png', nrow=6)
    return best_fid_this_param
