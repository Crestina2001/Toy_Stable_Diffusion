import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn, einsum
import torch.nn.functional as F
from model.__init__ import Unet
from dataLoader import get_data_loader
from torch.optim import Adam
from torchvision.utils import save_image
from pathlib import Path
from training import train_diffusion_model
import random
import logging
import datetime
import math
import copy


if __name__ == '__main__':
    logging.basicConfig(filename='training.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    random.seed(42)
    # declare image statistics for later use
    image_size = 28
    channels = 1
    # create dataloader
    batch_size = 128
    timesteps = 200  # NOTE: longer `timesteps` should be used with smaller `beta_end`
    beta_end = 0.08  # for example, for timesteps=1000, you can use beta_end=0.02
    num_epochs = 20
    beta_start=0.0001
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    params = {
        'image_size': image_size,
        'channels': channels,
        'batch_size': batch_size,
        'timesteps': timesteps,
        'beta_end': beta_end,
        'num_epochs': num_epochs,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'results_folder': results_folder
    }

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dataloader = get_data_loader(batch_size)
    # List of possible values for each parameter
    noise_schedules = ['linear', 'cosine', 'quadratic', 'sigmoid']
    timesteps_options = [(200, 0.08), (1000, 0.02)]  # (timesteps, beta_end)
    sampling_methods = ['DDPM', 'DDIM']
    # create the best_fid
    best_fid = float('inf')
    results = []  # List to store results
    # Iterating over all combinations
    for noise_schedule in noise_schedules:
        for timesteps, beta_end in timesteps_options:
            for sampling_method in sampling_methods:
                # Update params for the current combination
                params['noise_schedule'] = noise_schedule
                params['timesteps'] = timesteps
                params['beta_end'] = beta_end
                params['sampling_method'] = sampling_method

                print(f"Training with parameters: {params}")
                logging.info(f"Training with parameters: {params}")
                model = Unet(
                    dim=image_size,
                    channels=channels,
                    dim_mults=(1, 2, 4)
                )
                model.to(device)

                optimizer = Adam(model.parameters(), lr=1e-3)

                # Train the model with the current set of parameters
                fid = train_diffusion_model(model, dataloader, optimizer, params, device, best_fid)
                print(f'best result: {fid}')
                logging.info(f'best result: {fid}')
                if fid < best_fid:
                    best_fid = fid

                # Copy params and add fid to it
                result = copy.deepcopy(params)
                result['fid'] = fid
                results.append(result)
    print(results)
    # Dumping the results to a JSON file
    with open('grid_search_results.json', 'w') as file:
        json.dump(results, file, indent=4)
                



